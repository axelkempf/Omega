use std::path::PathBuf;
use std::process::Command;

use omega_indicators::{
    ATR, BollingerBands, EMA, GarchLocalParams, GarchVolatility, Indicator, KalmanGarchLocalParams,
    KalmanGarchZScore, KalmanZScore, MultiOutputIndicator, MultiTfIndicatorCache, ZScore,
    ZScoreMeanSource,
};
use omega_types::{Candle, PriceType, Timeframe};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct BollingerOutput {
    upper: Vec<Option<f64>>,
    middle: Vec<Option<f64>>,
    lower: Vec<Option<f64>>,
}

#[derive(Debug, Deserialize)]
struct PythonParityOutput {
    ema: Vec<Option<f64>>,
    ema_stepwise: Vec<Option<f64>>,
    bollinger_stepwise: BollingerOutput,
    atr: Vec<Option<f64>>,
    bollinger: BollingerOutput,
    zscore_rolling: Vec<Option<f64>>,
    zscore_ema: Vec<Option<f64>>,
    kalman_zscore: Vec<Option<f64>>,
    garch_volatility: Vec<Option<f64>>,
    kalman_garch_zscore: Vec<Option<f64>>,
    kalman_zscore_stepwise: Vec<Option<f64>>,
    garch_volatility_local: Vec<Option<f64>>,
    kalman_garch_zscore_local: Option<f64>,
}

#[test]
fn test_python_parity_stepwise_and_local() {
    let output = run_python_parity();

    let closes = [1.00, 1.10, 1.05, 1.20, 1.15, 1.30];
    let mut primary_bid = Vec::new();
    for (i, close) in closes.iter().enumerate() {
        primary_bid.push(make_candle(i as i64 * 60, *close));
    }
    let primary_ask = primary_bid.clone();

    let h1_closes = [1.05, 1.25];
    let mut h1_bid = Vec::new();
    for (i, close) in h1_closes.iter().enumerate() {
        h1_bid.push(make_candle(i as i64 * 180, *close));
    }
    let h1_ask = h1_bid.clone();

    let mut cache = MultiTfIndicatorCache::new(
        Timeframe::M1,
        primary_bid.clone(),
        primary_ask,
        vec![(Timeframe::H1, h1_bid, h1_ask)],
    );

    let ema = EMA::new(3).compute(&primary_bid);
    assert_series_close("ema", &output.ema, &ema, 1e-10);

    let ema_step = cache.ema_stepwise(Timeframe::H1, PriceType::Bid, 3);
    assert_series_close("ema_stepwise", &output.ema_stepwise, &ema_step, 1e-10);

    let (bb_upper, bb_middle, bb_lower) =
        cache.bollinger_stepwise(Timeframe::H1, PriceType::Bid, 3, 2.0);
    assert_series_close(
        "bollinger_stepwise.upper",
        &output.bollinger_stepwise.upper,
        &bb_upper,
        1e-10,
    );
    assert_series_close(
        "bollinger_stepwise.middle",
        &output.bollinger_stepwise.middle,
        &bb_middle,
        1e-10,
    );
    assert_series_close(
        "bollinger_stepwise.lower",
        &output.bollinger_stepwise.lower,
        &bb_lower,
        1e-10,
    );

    let kz_step = cache.kalman_zscore_stepwise(Timeframe::H1, PriceType::Bid, 3, 0.5, 0.1);
    assert_series_close(
        "kalman_zscore_stepwise",
        &output.kalman_zscore_stepwise,
        &kz_step,
        1e-10,
    );

    let garch_params = GarchLocalParams {
        alpha: 0.1,
        beta: 0.8,
        omega: None,
        use_log_returns: true,
        scale: 100.0,
        min_periods: 2,
        sigma_floor: 1e-6,
    };
    let garch_local =
        cache.garch_volatility_local(Timeframe::M1, PriceType::Bid, 5, 4, garch_params);
    assert_series_close(
        "garch_volatility_local",
        &output.garch_volatility_local,
        &garch_local,
        1e-10,
    );

    let kgz_params = KalmanGarchLocalParams {
        r: 0.01,
        q: 1.0,
        alpha: 0.1,
        beta: 0.8,
        omega: None,
        use_log_returns: true,
        scale: 100.0,
        min_periods: 2,
        sigma_floor: 1e-6,
    };
    let kgz_local =
        cache.kalman_garch_zscore_local(Timeframe::M1, PriceType::Bid, 5, 4, kgz_params);
    assert_scalar_close(
        "kalman_garch_zscore_local",
        output.kalman_garch_zscore_local,
        kgz_local,
        1e-10,
    );

    let atr = ATR::new(3).compute(&primary_bid);
    assert_series_close("atr", &output.atr, &atr, 1e-10);

    let bb = BollingerBands::new(3, 2.0).compute_all(&primary_bid);
    assert_series_close("bollinger.upper", &output.bollinger.upper, &bb.upper, 1e-10);
    assert_series_close(
        "bollinger.middle",
        &output.bollinger.middle,
        &bb.middle,
        1e-10,
    );
    assert_series_close("bollinger.lower", &output.bollinger.lower, &bb.lower, 1e-10);

    let zscore_rolling = ZScore::new(3).compute(&primary_bid);
    assert_series_close(
        "zscore_rolling",
        &output.zscore_rolling,
        &zscore_rolling,
        1e-10,
    );

    let zscore_ema =
        ZScore::with_mean_source(3, ZScoreMeanSource::Ema, Some(2)).compute(&primary_bid);
    assert_series_close("zscore_ema", &output.zscore_ema, &zscore_ema, 1e-10);

    let kalman_zscore = KalmanZScore::new(3, 0.5, 0.1).compute(&primary_bid);
    assert_series_close(
        "kalman_zscore",
        &output.kalman_zscore,
        &kalman_zscore,
        1e-10,
    );

    let garch_vol = GarchVolatility::new(0.1, 0.8, 0.0)
        .with_log_returns(true)
        .with_scale(100.0)
        .with_min_periods(2)
        .with_sigma_floor(1e-6)
        .compute(&primary_bid);
    assert_series_close(
        "garch_volatility",
        &output.garch_volatility,
        &garch_vol,
        1e-10,
    );

    let kgz = KalmanGarchZScore::new(0.01, 1.0, 0.1, 0.8, 0.0)
        .with_log_returns(true)
        .with_scale(100.0)
        .with_min_periods(2)
        .with_sigma_floor(1e-6)
        .compute(&primary_bid);
    assert_series_close(
        "kalman_garch_zscore",
        &output.kalman_garch_zscore,
        &kgz,
        1e-10,
    );
}

fn make_candle(timestamp_ns: i64, close: f64) -> Candle {
    Candle {
        timestamp_ns,
        open: close - 0.01,
        high: close + 0.02,
        low: close - 0.02,
        close,
        volume: 1.0,
    }
}

fn assert_series_close(label: &str, expected: &[Option<f64>], actual: &[f64], atol: f64) {
    assert_eq!(
        expected.len(),
        actual.len(),
        "{label}: length mismatch {} != {}",
        expected.len(),
        actual.len()
    );

    for (idx, (exp, act)) in expected.iter().zip(actual.iter()).enumerate() {
        match exp {
            None => assert!(!act.is_finite(), "{label}[{idx}] expected NaN, got {act}"),
            Some(value) => {
                assert!(act.is_finite(), "{label}[{idx}] expected finite, got {act}");
                let diff = (value - act).abs();
                assert!(diff <= atol, "{label}[{idx}] diff {diff} exceeds {atol}");
            }
        }
    }
}

fn assert_scalar_close(label: &str, expected: Option<f64>, actual: Option<f64>, atol: f64) {
    match (expected, actual) {
        (None, None) => {}
        (Some(exp), Some(act)) => {
            let diff = (exp - act).abs();
            assert!(
                diff <= atol,
                "{label} diff {diff} exceeds {atol}: {exp} vs {act}"
            );
        }
        (Some(exp), None) => panic!("{label} expected {exp}, got None"),
        (None, Some(act)) => panic!("{label} expected None, got {act}"),
    }
}

fn run_python_parity() -> PythonParityOutput {
    let python = std::env::var("PYTHON").unwrap_or_else(|_| "python3".to_string());
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let script_path = manifest_dir.join("tests").join("python_parity.py");

    let output = Command::new(&python)
        .arg(script_path)
        .output()
        .expect("Failed to run python parity script");

    if !output.status.success() {
        panic!(
            "Python parity script failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    serde_json::from_slice(&output.stdout).expect("Failed to parse python parity output")
}
