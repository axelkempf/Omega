use std::collections::HashMap;
use std::path::PathBuf;
use std::process::Command;

use omega_indicators::{IndicatorCache, IndicatorParams, IndicatorSpec, MultiTfIndicatorCache};
use omega_strategy::{BarContext, MeanReversionZScore, MrzParams, Strategy};
use omega_types::{Candle, Direction, Timeframe};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct ParityPayload {
    index: usize,
    params: serde_json::Value,
    data: ParityData,
    signals: HashMap<String, Option<SignalPayload>>,
    indicators: IndicatorSnapshot,
}

#[derive(Debug, Deserialize)]
struct ParitySuite {
    cases: HashMap<String, ParityPayload>,
}

#[derive(Debug, Deserialize)]
struct ParityData {
    bid_closes: Vec<f64>,
    ask_closes: Vec<f64>,
    h1_bid_closes: Vec<f64>,
    h1_ask_closes: Vec<f64>,
}

#[derive(Debug, Deserialize, Clone)]
struct SignalPayload {
    direction: String,
    entry: f64,
    sl: f64,
    tp: f64,
}

#[derive(Debug, Deserialize)]
struct IndicatorSnapshot {
    kalman_z: Option<f64>,
    kalman_garch_z: Option<f64>,
    bollinger: BollingerSnapshot,
    scenario6_debug: Scenario6Debug,
}

#[derive(Debug, Deserialize)]
struct Scenario6Debug {
    kalman_z_step: Option<f64>,
    lower: Option<f64>,
    price: Option<f64>,
    upper: Option<f64>,
    middle: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct BollingerSnapshot {
    upper: Option<f64>,
    middle: Option<f64>,
    lower: Option<f64>,
}

#[test]
fn test_mrz_python_parity_scenarios() {
    let suite = run_python_parity();

    for (case_key, direction) in [("long", "long"), ("short", "short")] {
        let payload = suite
            .cases
            .get(case_key)
            .unwrap_or_else(|| panic!("missing {case_key} payload"));
        let base_params: MrzParams =
            serde_json::from_value(payload.params.clone()).expect("invalid MRZ params");

        let primary_bid = build_candles(&payload.data.bid_closes, 0, 1);
        let primary_ask = build_candles(&payload.data.ask_closes, 0, 1);

        let h1_bid = build_candles(&payload.data.h1_bid_closes, 0, 4);
        let h1_ask = build_candles(&payload.data.h1_ask_closes, 0, 4);

        let mut indicator_cache = IndicatorCache::new();
        insert_kalman_z(
            &mut indicator_cache,
            payload.index,
            base_params.window_length,
            base_params.kalman_r,
            base_params.kalman_q,
            payload.indicators.kalman_z,
        );
        insert_kalman_garch_z(
            &mut indicator_cache,
            payload.index,
            base_params.window_length,
            KalmanGarchSnapshot {
                r: base_params.kalman_r,
                q: base_params.kalman_q,
                alpha: base_params.garch_alpha,
                beta: base_params.garch_beta,
                omega: base_params.garch_omega,
                use_log_returns: base_params.garch_use_log_returns,
                scale: base_params.garch_scale,
                min_periods: base_params.garch_min_periods,
                sigma_floor: base_params.garch_sigma_floor,
                value: payload.indicators.kalman_garch_z,
            },
        );
        insert_bollinger(
            &mut indicator_cache,
            payload.index,
            base_params.b_b_length,
            base_params.std_factor,
            &payload.indicators.bollinger,
        );

        insert_stepwise_kalman_z(
            &mut indicator_cache,
            payload.index,
            "H1",
            base_params.window_length,
            base_params.kalman_r,
            base_params.kalman_q,
            payload.indicators.scenario6_debug.kalman_z_step,
        );
        insert_stepwise_bollinger(
            &mut indicator_cache,
            payload.index,
            "H1",
            base_params.b_b_length,
            base_params.std_factor,
            &payload.indicators.scenario6_debug,
        );
        insert_tf_close(
            &mut indicator_cache,
            payload.index,
            "H1",
            payload.indicators.scenario6_debug.price,
        );

        let multi_tf = std::cell::RefCell::new(MultiTfIndicatorCache::new(
            Timeframe::M5,
            primary_bid.clone(),
            primary_ask.clone(),
            vec![(Timeframe::H1, h1_bid, h1_ask)],
        ));

        for scenario_id in 1..=6 {
            let mut params = base_params.clone();
            params.enabled_scenarios = vec![scenario_id];
            let mut strategy = MeanReversionZScore::new(params);

            let bid = &primary_bid[payload.index];
            let ask = &primary_ask[payload.index];
            let ctx = BarContext::new(payload.index, bid.timestamp_ns, bid, ask, &indicator_cache)
                .with_multi_tf(&multi_tf);

            let actual = strategy.on_bar(&ctx);
            let expected_key = format!("{scenario_id}_{direction}");
            let expected = payload.signals.get(&expected_key).cloned().unwrap_or(None);
            assert_signal_matches(case_key, scenario_id, expected, actual);
        }
    }
}

fn build_candles(closes: &[f64], start_ts: i64, step: i64) -> Vec<Candle> {
    closes
        .iter()
        .enumerate()
        .map(|(idx, close)| Candle {
            timestamp_ns: start_ts + (idx as i64 * step),
            open: close - 0.01,
            high: close + 0.02,
            low: close - 0.02,
            close: *close,
            volume: 1.0,
        })
        .collect()
}

fn insert_kalman_z(
    cache: &mut IndicatorCache,
    idx: usize,
    window: usize,
    r: f64,
    q: f64,
    value: Option<f64>,
) {
    let spec = IndicatorSpec::new(
        "KALMAN_Z",
        IndicatorParams::Kalman {
            window,
            r_x1000: (r * 1000.0).round() as u32,
            q_x1000: (q * 1000.0).round() as u32,
        },
    );
    insert_scalar(cache, spec, idx, value);
}

struct KalmanGarchSnapshot {
    r: f64,
    q: f64,
    alpha: f64,
    beta: f64,
    omega: f64,
    use_log_returns: bool,
    scale: f64,
    min_periods: usize,
    sigma_floor: f64,
    value: Option<f64>,
}

fn insert_kalman_garch_z(
    cache: &mut IndicatorCache,
    idx: usize,
    window: usize,
    snapshot: KalmanGarchSnapshot,
) {
    let spec = IndicatorSpec::new(
        "KALMAN_GARCH_Z",
        IndicatorParams::KalmanGarch {
            window,
            r_x1000: (snapshot.r * 1000.0).round() as u32,
            q_x1000: (snapshot.q * 1000.0).round() as u32,
            alpha_x1000: (snapshot.alpha * 1000.0).round() as u32,
            beta_x1000: (snapshot.beta * 1000.0).round() as u32,
            omega_x1000000: (snapshot.omega * 1_000_000.0).round() as u32,
            use_log_returns: snapshot.use_log_returns,
            scale_x100: (snapshot.scale * 100.0).round() as u32,
            min_periods: snapshot.min_periods,
            sigma_floor_x1e8: (snapshot.sigma_floor * 1e8).round() as u32,
        },
    );
    insert_scalar(cache, spec, idx, snapshot.value);
}

fn insert_bollinger(
    cache: &mut IndicatorCache,
    idx: usize,
    period: usize,
    std_factor: f64,
    snapshot: &BollingerSnapshot,
) {
    let spec = IndicatorSpec::new(
        "BOLLINGER",
        IndicatorParams::Bollinger {
            period,
            std_factor_x100: (std_factor * 100.0).round() as u32,
        },
    );
    insert_scalar(cache, spec.with_output_suffix("upper"), idx, snapshot.upper);
    insert_scalar(
        cache,
        spec.with_output_suffix("middle"),
        idx,
        snapshot.middle,
    );
    insert_scalar(cache, spec.with_output_suffix("lower"), idx, snapshot.lower);
}

fn insert_stepwise_kalman_z(
    cache: &mut IndicatorCache,
    idx: usize,
    timeframe: &str,
    window: usize,
    r: f64,
    q: f64,
    value: Option<f64>,
) {
    let spec = IndicatorSpec::new(
        format!("KALMAN_Z_{timeframe}").as_str(),
        IndicatorParams::Kalman {
            window,
            r_x1000: (r * 1000.0).round() as u32,
            q_x1000: (q * 1000.0).round() as u32,
        },
    );
    insert_scalar(cache, spec, idx, value);
}

fn insert_stepwise_bollinger(
    cache: &mut IndicatorCache,
    idx: usize,
    timeframe: &str,
    period: usize,
    std_factor: f64,
    snapshot: &Scenario6Debug,
) {
    let spec = IndicatorSpec::new(
        format!("BOLLINGER_{timeframe}").as_str(),
        IndicatorParams::Bollinger {
            period,
            std_factor_x100: (std_factor * 100.0).round() as u32,
        },
    );
    insert_scalar(cache, spec.with_output_suffix("upper"), idx, snapshot.upper);
    insert_scalar(
        cache,
        spec.with_output_suffix("middle"),
        idx,
        snapshot.middle,
    );
    insert_scalar(cache, spec.with_output_suffix("lower"), idx, snapshot.lower);
}

fn insert_tf_close(cache: &mut IndicatorCache, idx: usize, timeframe: &str, value: Option<f64>) {
    let spec = IndicatorSpec::new(
        format!("CLOSE_{timeframe}").as_str(),
        IndicatorParams::Period(1),
    );
    insert_scalar(cache, spec, idx, value);
}

fn insert_scalar(cache: &mut IndicatorCache, spec: IndicatorSpec, idx: usize, value: Option<f64>) {
    let mut values = vec![f64::NAN; idx + 1];
    if let Some(val) = value {
        values[idx] = val;
    }
    cache.insert(spec, values);
}

fn assert_signal_matches(
    case_key: &str,
    scenario_id: u8,
    expected: Option<SignalPayload>,
    actual: Option<omega_types::Signal>,
) {
    let tolerance = 1e-5;
    match (expected, actual) {
        (None, None) => {}
        (Some(exp), Some(act)) => {
            let expected_dir = match exp.direction.as_str() {
                "long" => Direction::Long,
                "short" => Direction::Short,
                other => panic!("Scenario {scenario_id}: unknown direction {other}"),
            };
            assert_eq!(
                act.direction, expected_dir,
                "Case {case_key} scenario {scenario_id} direction"
            );
            assert_close(
                exp.entry,
                act.entry_price,
                tolerance,
                case_key,
                scenario_id,
                "entry",
            );
            assert_close(
                exp.sl,
                act.stop_loss,
                tolerance,
                case_key,
                scenario_id,
                "sl",
            );
            assert_close(
                exp.tp,
                act.take_profit,
                tolerance,
                case_key,
                scenario_id,
                "tp",
            );
        }
        (None, Some(act)) => panic!(
            "Case {case_key} scenario {scenario_id}: expected no signal, got {:?}",
            act
        ),
        (Some(exp), None) => panic!(
            "Case {case_key} scenario {scenario_id}: expected signal {:?}, got None",
            exp
        ),
    }
}

fn assert_close(
    expected: f64,
    actual: f64,
    tol: f64,
    case_key: &str,
    scenario_id: u8,
    label: &str,
) {
    let diff = (expected - actual).abs();
    assert!(
        diff <= tol,
        "Case {case_key} scenario {scenario_id} {label} diff {diff} exceeds {tol}: {expected} vs {actual}"
    );
}

fn run_python_parity() -> ParitySuite {
    let python = std::env::var("PYTHON").unwrap_or_else(|_| "python3".to_string());
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let script_path = manifest_dir.join("tests").join("mrz_parity.py");

    let output = Command::new(&python)
        .arg(script_path)
        .output()
        .expect("Failed to run mrz_parity.py");

    if !output.status.success() {
        panic!(
            "mrz_parity.py failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    serde_json::from_slice(&output.stdout).expect("Failed to parse MRZ parity payload")
}
