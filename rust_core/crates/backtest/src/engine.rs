//! Backtest engine orchestration.

use std::cell::RefCell;
use std::collections::HashSet;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use omega_data::{
    CandleStore, MultiTfStore, filter_by_date_range, load_and_validate_bid_ask, resolve_data_path,
};
use omega_execution::{
    ExecutionCostsConfig, ExecutionEngine, ExecutionEngineConfig, FeeModel, NoFee, NoSlippage,
    SlippageModel, SymbolCosts, get_symbol_spec_or_default, load_symbol_specs,
};
use omega_indicators::{
    BollingerBands, IndicatorCache, IndicatorParams, IndicatorRegistry, IndicatorSpec,
    MultiOutputIndicator, MultiTfIndicatorCache, build_mapping,
};
use omega_portfolio::{
    Portfolio, calculate_gap_exit_price, check_stops,
};
use omega_strategy::{BarContext, HtfContext, IndicatorRequirement, Strategy, StrategyRegistry};
use omega_trade_mgmt::{
    Action, MarketView, PositionView, StopModifyReason, TradeContext, TradeManager,
    TradeManagerConfig, TradeManagerRulesConfig,
};
use omega_types::{
    BacktestConfig, BacktestResult, Candle, DataMode, Direction, RunMode, Signal, Timeframe,
};
use serde_json::Value as JsonValue;

use crate::context::RunContext;
use crate::error::BacktestError;
use crate::event_loop;
use crate::result_builder;
use crate::warmup::{validate_htf_warmup, validate_warmup};

/// Backtest engine orchestrating all components.
pub struct BacktestEngine {
    config: BacktestConfig,
    data: MultiTfStore,
    indicators: IndicatorCache,
    htf_indicators: Option<IndicatorCache>,
    execution: ExecutionEngine,
    symbol_costs: SymbolCosts,
    portfolio: Portfolio,
    strategy: Box<dyn Strategy>,
    trade_manager: TradeManager,
    run_ctx: RunContext,
    multi_tf_cache: RefCell<MultiTfIndicatorCache>,
    empty_htf_cache: IndicatorCache,
    pending_stop_updates: Vec<StopUpdate>,
    pending_tp_updates: Vec<TakeProfitUpdate>,
    pip_size: f64,
    pip_buffer_factor: f64,
    unit_value_per_price: f64,
    bar_duration_ns: i64,
    start_instant: Instant,
}

#[derive(Debug, Clone)]
struct StopUpdate {
    position_id: u64,
    new_stop_loss: f64,
    effective_from_idx: usize,
    reason: StopModifyReason,
}

#[derive(Debug, Clone)]
struct TakeProfitUpdate {
    position_id: u64,
    new_take_profit: f64,
    effective_from_idx: usize,
}

impl BacktestEngine {
    /// Creates a new backtest engine from config.
    ///
    /// # Errors
    /// Returns an error if the configuration is invalid, data loading fails,
    /// or indicator preparation cannot be completed.
    #[allow(clippy::too_many_lines)]
    pub fn new(config: BacktestConfig) -> Result<Self, BacktestError> {
        if config.data_mode != DataMode::Candle {
            return Err(BacktestError::ConfigValidation(
                "only candle data_mode is supported".to_string(),
            ));
        }

        let registry = StrategyRegistry::with_defaults();
        let strategy = registry.create(&config.strategy_name, &config.strategy_parameters)?;
        let required_htf_timeframes = strategy.required_htf_timeframes();

        let data = load_data(&config, &required_htf_timeframes)?;
        validate_warmup(&data.primary, config.warmup_bars)?;
        validate_htf_warmup(data.htf.as_ref(), config.warmup_bars)?;

        let primary_tf = data.primary.timeframe;
        let bar_duration_ns = timeframe_to_ns(primary_tf)?;

        let (indicators, htf_indicators) =
            compute_indicators(&data, &strategy.required_indicators(), data.htf.as_ref())?;

        // Validate indicators per DATA_FLOW_PLAN §5.2 Checkpoint 3
        validate_indicators(&indicators, data.primary.len(), config.warmup_bars)?;
        if let Some(ref htf_cache) = htf_indicators
            && let Some(ref htf_store) = data.htf
        {
            validate_indicators(htf_cache, htf_store.len(), config.warmup_bars)?;
        }

        let symbol_specs = load_symbol_specs(&resolve_symbol_specs_path())?;
        let symbol_spec = get_symbol_spec_or_default(&symbol_specs, &config.symbol);

        let costs_cfg = ExecutionCostsConfig::load(&resolve_execution_costs_path())?;
        let resolved_pip_size = config
            .costs
            .pip_size
            .unwrap_or_else(|| symbol_spec.resolved_pip_size());
        let mut symbol_costs = SymbolCosts::from_config(
            &costs_cfg,
            &config.symbol,
            resolved_pip_size,
            Some(&symbol_spec),
        );

        if config.costs.enabled {
            if (config.costs.slippage_multiplier - 1.0).abs() > f64::EPSILON {
                symbol_costs.slippage = Box::new(SlippageMultiplier::new(
                    symbol_costs.slippage,
                    config.costs.slippage_multiplier,
                ));
            }
            if (config.costs.fee_multiplier - 1.0).abs() > f64::EPSILON {
                symbol_costs.fee = Box::new(FeeMultiplier::new(
                    symbol_costs.fee,
                    config.costs.fee_multiplier,
                ));
            }
        } else {
            symbol_costs.slippage = Box::new(NoSlippage);
            symbol_costs.fee = Box::new(NoFee);
            symbol_costs.apply_entry_fee = false;
            symbol_costs.apply_exit_fee = false;
        }

        let rng_seed = match config.run_mode {
            RunMode::Dev => config.rng_seed.unwrap_or(42),
            RunMode::Prod => config.rng_seed.unwrap_or_else(random_seed),
        };

        let execution = ExecutionEngine::new(ExecutionEngineConfig {
            rng_seed,
            apply_entry_fees: config.costs.enabled,
            apply_exit_fees: config.costs.enabled,
            execution_variant: config.execution_variant,
        });

        let portfolio = Portfolio::new(
            config.account.initial_balance,
            config.account.max_positions,
            config.symbol.clone(),
        );

        let trade_manager = create_trade_manager(&config, bar_duration_ns);

        let run_ctx = RunContext::new(
            &data.primary.timestamps,
            config.sessions.as_deref(),
            config.news_filter.as_ref(),
            &config.symbol,
            bar_duration_ns,
        )?;

        let multi_tf_cache = RefCell::new(build_multi_tf_cache(&data));
        let pip_size = symbol_costs.pip_size;
        let pip_buffer_factor = config.costs.pip_buffer_factor;
        let unit_value_per_price = unit_value_per_price(&symbol_spec);

        Ok(Self {
            config,
            data,
            indicators,
            htf_indicators,
            execution,
            symbol_costs,
            portfolio,
            strategy,
            trade_manager,
            run_ctx,
            multi_tf_cache,
            empty_htf_cache: IndicatorCache::new(),
            pending_stop_updates: Vec::new(),
            pending_tp_updates: Vec::new(),
            pip_size,
            pip_buffer_factor,
            unit_value_per_price,
            bar_duration_ns,
            start_instant: Instant::now(),
        })
    }

    /// Runs the backtest event loop and returns the result.
    ///
    /// # Errors
    /// Returns an error if post-run consistency validation fails.
    pub fn run(mut self) -> Result<BacktestResult, BacktestError> {
        self.start_instant = Instant::now();
        event_loop::run_event_loop(&mut self);
        self.validate_portfolio_consistency()?;
        let runtime_seconds = self.runtime_seconds();
        let meta = result_builder::build_meta(
            &self.data.primary.timestamps,
            self.config.warmup_bars,
            runtime_seconds,
        );
        let fees_total = self.portfolio.total_fees();
        let risk_per_trade = self.config.account.risk_per_trade;
        let trades = self.portfolio.closed_trades().to_vec();
        let equity_curve = self.portfolio.into_equity_tracker().into_equity_curve();

        Ok(result_builder::build_result(
            trades,
            equity_curve,
            fees_total,
            risk_per_trade,
            meta,
        ))
    }

    pub(crate) fn warmup_bars(&self) -> usize {
        self.config.warmup_bars
    }

    pub(crate) fn primary_len(&self) -> usize {
        self.data.primary.len()
    }

    pub(crate) fn process_bar(&mut self, idx: usize) {
        let timestamp_ns = self.data.primary.timestamps[idx];
        let (bid, ask) = match self.data.primary.get(idx) {
            Some((bid, ask)) => (*bid, *ask),
            None => return,
        };

        self.apply_pending_updates(idx);

        let session_open = self.run_ctx.session_open(idx);
        let news_blocked = self.run_ctx.news_blocked(idx);

        self.check_pending_triggers(&bid, &ask, timestamp_ns);
        self.check_stops(&bid, &ask, timestamp_ns);
        self.apply_trade_management(
            idx,
            &bid,
            &ask,
            timestamp_ns,
            session_open,
            news_blocked,
        );

        let htf_ctx = Self::build_htf_context(
            &self.data,
            self.htf_indicators.as_ref(),
            &self.empty_htf_cache,
            idx,
        );
        let ctx = Self::build_context(
            idx,
            &bid,
            &ask,
            session_open,
            news_blocked,
            &self.indicators,
            &self.multi_tf_cache,
            htf_ctx,
        );

        if self.can_enter_position(&ctx)
            && let Some(signal) = self.strategy.on_bar(&ctx)
        {
            self.process_signal(signal, timestamp_ns);
        }

        let mid_price = f64::midpoint(bid.close, ask.close);
        self.portfolio.update_equity(timestamp_ns, mid_price);
    }

    fn can_enter_position(&self, ctx: &BarContext) -> bool {
        ctx.session_open && !ctx.news_blocked && self.portfolio.can_open_position()
    }

    #[allow(clippy::too_many_arguments)]
    fn build_context<'a>(
        idx: usize,
        bid: &'a Candle,
        ask: &'a Candle,
        session_open: bool,
        news_blocked: bool,
        indicators: &'a IndicatorCache,
        multi_tf_cache: &'a RefCell<MultiTfIndicatorCache>,
        htf_ctx: Option<HtfContext<'a>>,
    ) -> BarContext<'a> {
        let mut ctx = BarContext::new(idx, bid.timestamp_ns, bid, ask, indicators)
            .with_multi_tf(multi_tf_cache)
            .with_session(session_open)
            .with_news_blocked(news_blocked);

        if let Some(htf_ctx) = htf_ctx {
            ctx = ctx.with_htf(htf_ctx);
        }

        ctx
    }

    /// Builds HTF context using only **completed** HTF bars (Lookahead Prevention).
    ///
    /// Per `DATA_FLOW_PLAN` §4.2:
    /// - `completed_idx = htf_idx - 1` (only use the last **completed** HTF bar)
    /// - If `htf_idx == 0` (no completed bar yet), return `None`
    ///
    /// This prevents lookahead bias where strategies would see HTF data
    /// from the current, still-forming HTF bar.
    fn build_htf_context<'a>(
        data: &'a MultiTfStore,
        htf_indicators: Option<&'a IndicatorCache>,
        empty_htf_cache: &'a IndicatorCache,
        idx: usize,
    ) -> Option<HtfContext<'a>> {
        let htf_store = data.htf.as_ref()?;
        let htf_idx = data.htf_index_at(idx)?;

        // Lookahead Prevention: Only use completed HTF bars.
        // htf_idx points to the current/forming HTF bar, so we need htf_idx - 1.
        // If htf_idx == 0, there's no completed bar yet → return None.
        let completed_idx = htf_idx.checked_sub(1)?;

        let (bid, ask) = htf_store.get(completed_idx)?;
        let cache = htf_indicators.unwrap_or(empty_htf_cache);
        Some(HtfContext::new(
            bid,
            ask,
            cache,
            completed_idx,
            htf_store.timeframe.as_str(),
        ))
    }

    fn check_pending_triggers(&mut self, bid: &Candle, ask: &Candle, timestamp_ns: i64) {
        let result = self
            .execution
            .process_pending_orders(bid, ask, timestamp_ns, &self.symbol_costs);

        for fill in result.fills {
            let entry_time_ns = fill.order.triggered_at_ns.unwrap_or(timestamp_ns);
            let signal = Signal {
                direction: fill.order.direction,
                order_type: fill.order.order_type,
                entry_price: fill.order.entry_price,
                stop_loss: fill.order.stop_loss,
                take_profit: fill.order.take_profit,
                size: Some(fill.order.size),
                scenario_id: fill.order.scenario_id,
                tags: Vec::new(),
                meta: fill.order.meta.clone(),
            };

            if let Err(err) = self.portfolio.open_position(
                &signal,
                fill.fill_price,
                fill.order.size,
                entry_time_ns,
                fill.entry_fee,
            ) {
                tracing::warn!("pending fill rejected by portfolio: {err}");
            }
        }

        if !result.rejected.is_empty() {
            for rejection in result.rejected {
                tracing::warn!(
                    "pending order {} rejected: {}",
                    rejection.order_id,
                    rejection.reason
                );
            }
        }

        self.execution.cleanup();
    }

    fn check_stops(&mut self, bid: &Candle, ask: &Candle, timestamp_ns: i64) {
        let pip_size = self.pip_size;
        let pip_buffer = self.pip_buffer_factor;

        let positions_to_close: Vec<_> = self
            .portfolio
            .positions()
            .iter()
            .filter_map(|pos| {
                let in_entry_candle = pos.entry_time_ns == timestamp_ns;
                check_stops(pos, bid, ask, pip_size, pip_buffer, in_entry_candle).map(|result| {
                    (
                        pos.id,
                        pos.direction,
                        pos.size,
                        result,
                    )
                })
            })
            .collect();

        for (pos_id, direction, size, result) in positions_to_close {
            let is_stop_loss = matches!(result.reason, omega_types::ExitReason::StopLoss);
            let gap_exit = calculate_gap_exit_price(result.exit_price, &direction, bid, ask, is_stop_loss);
            let exit_fill = self
                .execution
                .apply_exit_slippage(gap_exit, direction, size, &self.symbol_costs);

            self.portfolio.close_position(
                pos_id,
                exit_fill.fill_price,
                timestamp_ns,
                result.reason,
                exit_fill.exit_fee,
            );
        }
    }

    fn apply_trade_management(
        &mut self,
        idx: usize,
        bid: &Candle,
        ask: &Candle,
        timestamp_ns: i64,
        session_open: bool,
        news_blocked: bool,
    ) {
        if !self.trade_manager.has_rules() {
            return;
        }

        let market = MarketView {
            timestamp_ns,
            bid_open: bid.open,
            bid_high: bid.high,
            bid_low: bid.low,
            bid_close: bid.close,
            ask_open: ask.open,
            ask_high: ask.high,
            ask_low: ask.low,
            ask_close: ask.close,
        };
        let ctx = TradeContext::new(idx, market, self.bar_duration_ns)
            .with_session(session_open)
            .with_news_blocked(news_blocked);

        let positions: Vec<PositionView> = self
            .portfolio
            .positions()
            .iter()
            .map(|pos| {
                PositionView::new(
                    pos.id,
                    self.config.symbol.clone(),
                    pos.direction,
                    pos.entry_time_ns,
                    pos.entry_price,
                    pos.size,
                )
                .with_stop_loss(pos.stop_loss)
                .with_take_profit(pos.take_profit)
                .with_scenario(pos.scenario_id)
                .with_meta(pos.meta.clone())
            })
            .collect();

        let actions = self.trade_manager.evaluate(&ctx, &positions);

        for action in actions {
            match action {
                Action::ClosePosition {
                    position_id,
                    reason,
                    exit_price_hint,
                    ..
                } => {
                    let Some(position) = self
                        .portfolio
                        .positions()
                        .iter()
                        .find(|pos| pos.id == position_id)
                    else {
                        continue;
                    };

                    let exit_price = exit_price_hint.unwrap_or_else(|| market.exit_price(position.direction));
                    let exit_fill = self.execution.apply_exit_slippage(
                        exit_price,
                        position.direction,
                        position.size,
                        &self.symbol_costs,
                    );

                    self.portfolio.close_position(
                        position_id,
                        exit_fill.fill_price,
                        timestamp_ns,
                        reason,
                        exit_fill.exit_fee,
                    );
                }
                Action::ModifyStopLoss {
                    position_id,
                    new_stop_loss,
                    reason,
                    effective_from_idx,
                } => {
                    self.pending_stop_updates.push(StopUpdate {
                        position_id,
                        new_stop_loss,
                        effective_from_idx,
                        reason,
                    });
                }
                Action::ModifyTakeProfit {
                    position_id,
                    new_take_profit,
                    effective_from_idx,
                } => {
                    self.pending_tp_updates.push(TakeProfitUpdate {
                        position_id,
                        new_take_profit,
                        effective_from_idx,
                    });
                }
                Action::None => {}
            }
        }
    }

    fn process_signal(&mut self, mut signal: Signal, timestamp_ns: i64) {
        if signal.size.is_none()
            && let Some(size) = self.calculate_position_size(&signal)
        {
            signal.size = Some(size);
        }

        match signal.order_type {
            omega_types::OrderType::Market => {
                match self
                    .execution
                    .execute_market_order(&signal, &self.symbol_costs)
                {
                    Ok(fill) => {
                        if fill.filled
                            && let Err(err) = self.portfolio.open_position(
                                &signal,
                                fill.fill_price,
                                fill.size,
                                timestamp_ns,
                                fill.entry_fee,
                            )
                        {
                            tracing::warn!("market order rejected by portfolio: {err}");
                        }
                    }
                    Err(err) => {
                        tracing::warn!("market order rejected: {err}");
                    }
                }
            }
            omega_types::OrderType::Limit | omega_types::OrderType::Stop => {
                if let Err(err) = self
                    .execution
                    .add_pending_order(&signal, timestamp_ns, &self.symbol_costs)
                {
                    tracing::warn!("pending order rejected: {err}");
                }
            }
        }
    }

    fn calculate_position_size(&self, signal: &Signal) -> Option<f64> {
        let risk_per_trade = self.config.account.risk_per_trade;
        let sl_distance = (signal.entry_price - signal.stop_loss).abs();
        if !risk_per_trade.is_finite() || risk_per_trade <= 0.0 {
            return None;
        }
        if !sl_distance.is_finite() || sl_distance <= 0.0 {
            return None;
        }
        let risk_per_lot = sl_distance * self.unit_value_per_price;
        if risk_per_lot <= 0.0 || !risk_per_lot.is_finite() {
            return None;
        }
        Some(risk_per_trade / risk_per_lot)
    }

    fn apply_pending_updates(&mut self, idx: usize) {
        if !self.pending_stop_updates.is_empty() {
            let mut remaining = Vec::new();
            let mut applied = Vec::new();
            for update in self.pending_stop_updates.drain(..) {
                if update.effective_from_idx > idx {
                    remaining.push(update);
                    continue;
                }
                if let Err(err) = self
                    .portfolio
                    .position_manager_mut()
                    .modify_stop_loss(update.position_id, update.new_stop_loss)
                {
                    tracing::warn!("stop update rejected: {err}");
                } else {
                    applied.push((update.position_id, update.reason));
                }
            }
            self.pending_stop_updates = remaining;
            for (position_id, reason) in applied {
                self.annotate_stop_kind(position_id, reason);
            }
        }

        if !self.pending_tp_updates.is_empty() {
            let mut remaining = Vec::new();
            for update in self.pending_tp_updates.drain(..) {
                if update.effective_from_idx > idx {
                    remaining.push(update);
                    continue;
                }
                if let Err(err) = self
                    .portfolio
                    .position_manager_mut()
                    .modify_take_profit(update.position_id, update.new_take_profit)
                {
                    tracing::warn!("take-profit update rejected: {err}");
                }
            }
            self.pending_tp_updates = remaining;
        }
    }

    fn annotate_stop_kind(&mut self, position_id: u64, reason: StopModifyReason) {
        let Some(position) = self
            .portfolio
            .position_manager_mut()
            .get_position_mut(position_id)
        else {
            return;
        };

        let kind = match reason {
            StopModifyReason::BreakEven => "break_even",
            StopModifyReason::Trailing => "trailing",
            StopModifyReason::Manual => "manual",
        };

        match &mut position.meta {
            JsonValue::Object(map) => {
                map.insert(
                    "stop_loss_kind".to_string(),
                    JsonValue::String(kind.to_string()),
                );
            }
            other => {
                let mut map = serde_json::Map::new();
                if !other.is_null() {
                    map.insert("raw_meta".to_string(), other.clone());
                }
                map.insert(
                    "stop_loss_kind".to_string(),
                    JsonValue::String(kind.to_string()),
                );
                position.meta = JsonValue::Object(map);
            }
        }
    }

    pub(crate) fn runtime_seconds(&self) -> f64 {
        self.start_instant.elapsed().as_secs_f64()
    }

    fn validate_portfolio_consistency(&self) -> Result<(), BacktestError> {
        let len = self.data.primary.len();
        if len == 0 {
            return Err(BacktestError::Runtime(
                "no candle data for portfolio consistency check".to_string(),
            ));
        }

        let last_idx = len - 1;
        let (bid, ask) = self.data.primary.get(last_idx).ok_or_else(|| {
            BacktestError::Runtime("missing last candle for portfolio consistency check".to_string())
        })?;
        let mid_price = f64::midpoint(bid.close, ask.close);

        self.portfolio.validate_consistency(mid_price)?;
        Ok(())
    }
}

fn load_data(config: &BacktestConfig, required_tfs: &[String]) -> Result<MultiTfStore, BacktestError> {
    let primary_tf = parse_timeframe(&config.timeframes.primary)?;
    let mut additional: Vec<String> = Vec::new();
    additional.extend(config.timeframes.additional.iter().cloned());
    additional.extend(required_tfs.iter().cloned());

    let htf_name = required_tfs.first().map(|tf| tf.trim().to_uppercase());

    let mut seen = HashSet::new();
    let mut additional_tfs = Vec::new();
    for tf in additional {
        let norm = tf.trim().to_uppercase();
        if norm.is_empty() {
            continue;
        }
        if norm == primary_tf.as_str() {
            continue;
        }
        if htf_name.as_deref().is_some_and(|htf| htf == norm) {
            continue;
        }
        if seen.insert(norm.clone()) {
            additional_tfs.push(norm);
        }
    }

    let start_ns = parse_datetime_ns(&config.start_date, DateBoundary::Start)?;
    let end_ns = parse_datetime_ns(&config.end_date, DateBoundary::End)?;
    if start_ns > end_ns {
        return Err(BacktestError::ConfigValidation(
            "start_date must be before end_date".to_string(),
        ));
    }

    let max_tf_seconds = max_timeframe_seconds(primary_tf, &additional_tfs, htf_name.as_deref())?;
    let warmup_delta = warmup_delta_ns(config.warmup_bars, max_tf_seconds)?;
    let extended_start_ns = start_ns.saturating_sub(warmup_delta);

    let primary = load_timeframe_store(
        &config.symbol,
        primary_tf,
        extended_start_ns,
        end_ns,
        config.warmup_bars,
    )?;

    let htf = if let Some(htf) = htf_name {
        let tf = parse_timeframe(&htf)?;
        Some(load_timeframe_store(
            &config.symbol,
            tf,
            extended_start_ns,
            end_ns,
            config.warmup_bars,
        )?)
    } else {
        None
    };

    let mut additional_stores = Vec::new();
    for tf_name in additional_tfs {
        let tf = parse_timeframe(&tf_name)?;
        let store = load_timeframe_store(
            &config.symbol,
            tf,
            extended_start_ns,
            end_ns,
            config.warmup_bars,
        )?;
        additional_stores.push(store);
    }

    Ok(MultiTfStore::new(primary, htf, additional_stores))
}

fn load_timeframe_store(
    symbol: &str,
    timeframe: Timeframe,
    start_ns: i64,
    end_ns: i64,
    warmup_bars: usize,
) -> Result<CandleStore, BacktestError> {
    let bid_path = resolve_data_path(symbol, timeframe.as_str(), "BID");
    let ask_path = resolve_data_path(symbol, timeframe.as_str(), "ASK");
    let aligned = load_and_validate_bid_ask(&bid_path, &ask_path, timeframe)?;

    let bid = filter_by_date_range(&aligned.bid, start_ns, end_ns)?;
    let ask = filter_by_date_range(&aligned.ask, start_ns, end_ns)?;

    if bid.len() != ask.len() {
        return Err(BacktestError::Data(
            omega_data::DataError::AlignmentFailure(
                "bid/ask length mismatch after date filter".to_string(),
            ),
        ));
    }

    let timestamps: Vec<i64> = bid.iter().map(|c| c.timestamp_ns).collect();

    Ok(CandleStore {
        bid,
        ask,
        timestamps,
        timeframe,
        symbol: symbol.to_string(),
        warmup_bars,
    })
}

#[allow(clippy::too_many_lines)]
fn compute_indicators(
    data: &MultiTfStore,
    requirements: &[IndicatorRequirement],
    htf_store: Option<&CandleStore>,
) -> Result<(IndicatorCache, Option<IndicatorCache>), BacktestError> {
    let primary_tf = data.primary.timeframe;
    let mut cache = IndicatorCache::new();
    let mut htf_cache = htf_store.map(|_| IndicatorCache::new());
    let registry = IndicatorRegistry::with_defaults();

    for req in requirements {
        let spec = indicator_spec_from_params(&req.name, &req.params)
            .ok_or_else(|| BacktestError::Indicator(omega_indicators::IndicatorError::InvalidParams(
                format!("invalid indicator params for {}", req.name),
            )))?;

        let target_tf = if let Some(tf) = req.timeframe.as_deref() {
            parse_timeframe(tf)?
        } else {
            primary_tf
        };

        let target_store = find_store(data, target_tf)
            .ok_or_else(|| BacktestError::InvalidTimeframe(target_tf.as_str().to_string()))?;

        let needs_primary = !cache.contains(&spec);
        let needs_htf = htf_store
            .is_some_and(|store| store.timeframe == target_tf)
            && htf_cache
                .as_ref()
                .is_some_and(|htf_cache| !htf_cache.contains(&spec));

        if !needs_primary && !needs_htf {
            continue;
        }

        let base_name = base_indicator_name(&req.name);
        match base_name.as_str() {
            "BOLLINGER" => {
                let period = req
                    .params
                    .get("period")
                    .and_then(JsonValue::as_u64)
                    .and_then(u64_to_usize)
                    .ok_or_else(|| {
                        BacktestError::Indicator(omega_indicators::IndicatorError::InvalidParams(
                            "BOLLINGER requires period".to_string(),
                        ))
                    })?;
                let std_factor = req
                    .params
                    .get("std_factor")
                    .and_then(JsonValue::as_f64)
                    .unwrap_or(2.0);

                let bb = BollingerBands::new(period, std_factor);
                let result = bb.compute_all(&target_store.bid);

                insert_bollinger_outputs(
                    &mut cache,
                    &spec,
                    &result.upper,
                    &result.middle,
                    &result.lower,
                    data,
                    target_store,
                    needs_primary,
                );

                if let Some(htf_cache) = &mut htf_cache
                    && needs_htf
                {
                    htf_cache.insert(spec.with_output_suffix("upper"), result.upper.clone());
                    htf_cache.insert(spec.with_output_suffix("middle"), result.middle.clone());
                    htf_cache.insert(spec.with_output_suffix("lower"), result.lower.clone());
                }
            }
            "CLOSE" => {
                let values: Vec<f64> = target_store.bid.iter().map(|c| c.close).collect();
                if needs_primary {
                    let aligned = if target_tf == primary_tf {
                        values.clone()
                    } else {
                        align_to_primary(&data.primary.timestamps, target_store, &values)
                    };
                    cache.insert(spec.clone(), aligned);
                }
                if let Some(htf_cache) = &mut htf_cache
                    && needs_htf
                {
                    htf_cache.insert(spec.clone(), values);
                }
            }
            _ => {
                let base_spec = IndicatorSpec::new(base_name.clone(), spec.params.clone());
                let indicator = registry.create(&base_spec)?;
                let values = indicator.compute(&target_store.bid);

                if needs_primary {
                    let aligned = if target_tf == primary_tf {
                        values.clone()
                    } else {
                        align_to_primary(&data.primary.timestamps, target_store, &values)
                    };
                    cache.insert(spec.clone(), aligned);
                }
                if let Some(htf_cache) = &mut htf_cache
                    && needs_htf
                {
                    htf_cache.insert(spec.clone(), values);
                }
            }
        }
    }

    Ok((cache, htf_cache))
}

    #[allow(clippy::too_many_arguments)]
    fn insert_bollinger_outputs(
    cache: &mut IndicatorCache,
    spec: &IndicatorSpec,
    upper: &[f64],
    middle: &[f64],
    lower: &[f64],
    data: &MultiTfStore,
    target_store: &CandleStore,
    needs_primary: bool,
) {
    if !needs_primary {
        return;
    }

    let upper_aligned = if target_store.timeframe == data.primary.timeframe {
        upper.to_vec()
    } else {
        align_to_primary(&data.primary.timestamps, target_store, upper)
    };
    let middle_aligned = if target_store.timeframe == data.primary.timeframe {
        middle.to_vec()
    } else {
        align_to_primary(&data.primary.timestamps, target_store, middle)
    };
    let lower_aligned = if target_store.timeframe == data.primary.timeframe {
        lower.to_vec()
    } else {
        align_to_primary(&data.primary.timestamps, target_store, lower)
    };

    cache.insert(spec.with_output_suffix("upper"), upper_aligned);
    cache.insert(spec.with_output_suffix("middle"), middle_aligned);
    cache.insert(spec.with_output_suffix("lower"), lower_aligned);
}

fn align_to_primary(
    primary_timestamps: &[i64],
    target_store: &CandleStore,
    values: &[f64],
) -> Vec<f64> {
    let mapping = build_mapping(
        primary_timestamps,
        &target_store.timestamps,
        target_store.timeframe,
    );

    let mut expanded = vec![f64::NAN; primary_timestamps.len()];
    let mut last_target = None;

    for (idx, maybe_target) in mapping.primary_to_target.iter().enumerate() {
        if let Some(target_idx) = *maybe_target {
            if last_target != Some(target_idx) {
                expanded[idx] = values.get(target_idx).copied().unwrap_or(f64::NAN);
                last_target = Some(target_idx);
            } else if idx > 0 {
                expanded[idx] = expanded[idx - 1];
            }
        }
    }

    expanded
}

fn find_store(data: &MultiTfStore, tf: Timeframe) -> Option<&CandleStore> {
    if data.primary.timeframe == tf {
        return Some(&data.primary);
    }
    if let Some(htf) = &data.htf
        && htf.timeframe == tf
    {
        return Some(htf);
    }
    data.additional.iter().find(|store| store.timeframe == tf)
}

fn build_multi_tf_cache(store: &MultiTfStore) -> MultiTfIndicatorCache {
    let primary_tf = store.primary.timeframe;
    let mut additional = Vec::new();
    let mut seen = HashSet::new();

    if let Some(htf) = &store.htf
        && htf.timeframe != primary_tf
        && seen.insert(htf.timeframe)
    {
        additional.push((htf.timeframe, htf.bid.clone(), htf.ask.clone()));
    }

    for extra in &store.additional {
        if extra.timeframe != primary_tf && seen.insert(extra.timeframe) {
            additional.push((extra.timeframe, extra.bid.clone(), extra.ask.clone()));
        }
    }

    MultiTfIndicatorCache::new(
        primary_tf,
        store.primary.bid.clone(),
        store.primary.ask.clone(),
        additional,
    )
}

fn create_trade_manager(config: &BacktestConfig, bar_duration_ns: i64) -> TradeManager {
    let Some(tm_cfg) = config.trade_management.as_ref() else {
        return TradeManager::empty();
    };

    let max_minutes = tm_cfg
        .rules
        .max_holding_time
        .max_holding_minutes
        .unwrap_or(0);
    let trade_manager_config = TradeManagerConfig {
        enabled: tm_cfg.enabled,
        stop_update_policy: match tm_cfg.stop_update_policy {
            omega_types::StopUpdatePolicy::ApplyNextBar => {
                omega_trade_mgmt::StopUpdatePolicy::ApplyNextBar
            }
        },
        rules: TradeManagerRulesConfig {
            max_holding_time: omega_trade_mgmt::MaxHoldingTimeConfig {
                enabled: tm_cfg.rules.max_holding_time.enabled,
                max_holding_minutes: max_minutes,
                only_scenarios: tm_cfg.rules.max_holding_time.only_scenarios.clone(),
            },
        },
    };

    let fallback_max = config
        .strategy_parameters
        .get("max_holding_minutes")
        .and_then(JsonValue::as_u64);

    TradeManager::from_config(&trade_manager_config, bar_duration_ns, fallback_max)
}

fn parse_timeframe(value: &str) -> Result<Timeframe, BacktestError> {
    value
        .parse::<Timeframe>()
        .map_err(|_| BacktestError::InvalidTimeframe(value.to_string()))
}

fn max_timeframe_seconds(
    primary: Timeframe,
    additional: &[String],
    htf: Option<&str>,
) -> Result<u64, BacktestError> {
    let mut max_secs = primary.to_seconds();

    if let Some(htf_name) = htf {
        let tf = parse_timeframe(htf_name)?;
        max_secs = max_secs.max(tf.to_seconds());
    }

    for tf_name in additional {
        let tf = parse_timeframe(tf_name)?;
        max_secs = max_secs.max(tf.to_seconds());
    }

    Ok(max_secs)
}

fn warmup_delta_ns(warmup_bars: usize, tf_seconds: u64) -> Result<i64, BacktestError> {
    let bars = u128::from(u64_from_usize(warmup_bars));
    let seconds = bars.saturating_mul(u128::from(tf_seconds));
    let total_ns = seconds.saturating_mul(1_000_000_000u128);
    i64::try_from(total_ns).map_err(|_| {
        BacktestError::ConfigValidation("warmup period too large".to_string())
    })
}

fn timeframe_to_ns(tf: Timeframe) -> Result<i64, BacktestError> {
    let seconds = tf.to_seconds();
    let ns = seconds
        .checked_mul(1_000_000_000)
        .ok_or_else(|| BacktestError::Runtime("timeframe overflow".to_string()))?;
    i64::try_from(ns).map_err(|_| BacktestError::Runtime("timeframe overflow".to_string()))
}

/// Validates indicator arrays per `DATA_FLOW_PLAN` §5.1 Invariants I4 and I5.
///
/// Checks:
/// - I4: All indicator arrays have length == `expected_len`
/// - I5: No NaN values after warmup period (from index `warmup_bars` onwards)
fn validate_indicators(
    cache: &IndicatorCache,
    expected_len: usize,
    warmup_bars: usize,
) -> Result<(), BacktestError> {
    for spec in cache.specs() {
        let Some(values) = cache.get(spec) else {
            continue;
        };

        // I4: Length must match candle count
        if values.len() != expected_len {
            return Err(BacktestError::Runtime(format!(
                "indicator '{}' length mismatch: expected {}, got {}",
                spec.name,
                expected_len,
                values.len()
            )));
        }

        // I5: No NaN after warmup period
        for (idx, &value) in values.iter().enumerate().skip(warmup_bars) {
            if value.is_nan() {
                return Err(BacktestError::Runtime(format!(
                    "indicator '{}' has NaN at index {} (after warmup period {})",
                    spec.name, idx, warmup_bars
                )));
            }
        }
    }
    Ok(())
}

#[allow(clippy::too_many_lines)]
fn indicator_spec_from_params(name: &str, params: &JsonValue) -> Option<IndicatorSpec> {
    let name_upper = name.trim().to_uppercase();
    let base_name = base_indicator_name(&name_upper);

    let indicator_params = match base_name.as_str() {
        "EMA" | "SMA" | "ATR" | "Z_SCORE" => {
            let period = params.get("period").or_else(|| params.get("window"))?;
            let period = period.as_u64().and_then(u64_to_usize)?;
            IndicatorParams::Period(period)
        }
        "BOLLINGER" => {
            let period = params
                .get("period")
                .and_then(JsonValue::as_u64)
                .and_then(u64_to_usize)?;
            let std_factor = params
                .get("std_factor")
                .and_then(JsonValue::as_f64)
                .unwrap_or(2.0);
            IndicatorParams::Bollinger {
                period,
                std_factor_x100: scale_to_u32(std_factor, 100.0),
            }
        }
        "KALMAN_Z" | "KALMAN_ZSCORE" => {
            let window = params
                .get("window")
                .and_then(JsonValue::as_u64)
                .and_then(u64_to_usize)?;
            let r = params.get("r").and_then(JsonValue::as_f64).unwrap_or(1.0);
            let q = params.get("q").and_then(JsonValue::as_f64).unwrap_or(0.1);
            IndicatorParams::Kalman {
                window,
                r_x1000: scale_to_u32(r, 1000.0),
                q_x1000: scale_to_u32(q, 1000.0),
            }
        }
        "GARCH_VOL" | "GARCH" => {
            let alpha = params
                .get("alpha")
                .and_then(JsonValue::as_f64)
                .unwrap_or(0.1);
            let beta = params
                .get("beta")
                .and_then(JsonValue::as_f64)
                .unwrap_or(0.8);
            let omega = params
                .get("omega")
                .and_then(JsonValue::as_f64)
                .unwrap_or(0.00001);
            let use_log_returns = params
                .get("use_log_returns")
                .and_then(JsonValue::as_bool)
                .unwrap_or(true);
            let scale = params
                .get("scale")
                .and_then(JsonValue::as_f64)
                .unwrap_or(100.0);
            let min_periods = params
                .get("min_periods")
                .and_then(JsonValue::as_u64)
                .and_then(u64_to_usize)
                .unwrap_or(20);
            let sigma_floor = params
                .get("sigma_floor")
                .and_then(JsonValue::as_f64)
                .unwrap_or(0.0001);
            IndicatorParams::Garch {
                alpha_x1000: scale_to_u32(alpha, 1000.0),
                beta_x1000: scale_to_u32(beta, 1000.0),
                omega_x1000000: scale_to_u32(omega, 1_000_000.0),
                use_log_returns,
                scale_x100: scale_to_u32(scale, 100.0),
                min_periods,
                sigma_floor_x1e8: scale_to_u32(sigma_floor, 1e8),
            }
        }
        "KALMAN_GARCH" | "KALMAN_GARCH_Z" => {
            let window = params
                .get("window")
                .and_then(JsonValue::as_u64)
                .and_then(u64_to_usize)?;
            let r = params.get("r").and_then(JsonValue::as_f64).unwrap_or(1.0);
            let q = params.get("q").and_then(JsonValue::as_f64).unwrap_or(0.1);
            let alpha = params
                .get("alpha")
                .and_then(JsonValue::as_f64)
                .unwrap_or(0.1);
            let beta = params
                .get("beta")
                .and_then(JsonValue::as_f64)
                .unwrap_or(0.8);
            let omega = params
                .get("omega")
                .and_then(JsonValue::as_f64)
                .unwrap_or(0.00001);
            let use_log_returns = params
                .get("use_log_returns")
                .and_then(JsonValue::as_bool)
                .unwrap_or(true);
            let scale = params
                .get("scale")
                .and_then(JsonValue::as_f64)
                .unwrap_or(1.0);
            let min_periods = params
                .get("min_periods")
                .and_then(JsonValue::as_u64)
                .and_then(u64_to_usize)
                .unwrap_or(20);
            let sigma_floor = params
                .get("sigma_floor")
                .and_then(JsonValue::as_f64)
                .unwrap_or(1e-8);
            IndicatorParams::KalmanGarch {
                window,
                r_x1000: scale_to_u32(r, 1000.0),
                q_x1000: scale_to_u32(q, 1000.0),
                alpha_x1000: scale_to_u32(alpha, 1000.0),
                beta_x1000: scale_to_u32(beta, 1000.0),
                omega_x1000000: scale_to_u32(omega, 1_000_000.0),
                use_log_returns,
                scale_x100: scale_to_u32(scale, 100.0),
                min_periods,
                sigma_floor_x1e8: scale_to_u32(sigma_floor, 1e8),
            }
        }
        "VOL_CLUSTER" => {
            let vol_period = params.get("window").or_else(|| params.get("vol_period"))?;
            let high_thresh = params
                .get("high_vol_threshold")
                .and_then(JsonValue::as_f64)
                .unwrap_or(1.5);
            let low_thresh = params
                .get("low_vol_threshold")
                .and_then(JsonValue::as_f64)
                .unwrap_or(0.5);
            let vol_period = vol_period.as_u64().and_then(u64_to_usize)?;
            IndicatorParams::VolCluster {
                vol_period,
                high_vol_threshold_x100: scale_to_u32(high_thresh, 100.0),
                low_vol_threshold_x100: scale_to_u32(low_thresh, 100.0),
            }
        }
        "CLOSE" => IndicatorParams::Period(1),
        _ => {
            if let Some(period) = params.get("period").or_else(|| params.get("window")) {
                let period = period.as_u64().and_then(u64_to_usize)?;
                IndicatorParams::Period(period)
            } else {
                return None;
            }
        }
    };

    Some(IndicatorSpec::new(name_upper, indicator_params))
}

fn base_indicator_name(name: &str) -> String {
    let name_upper = name.to_uppercase();
    name_upper
        .rsplit_once('_')
        .and_then(|(base, suffix)| {
            let bytes = suffix.as_bytes();
            if bytes.len() < 2 {
                return None;
            }
            let prefix = bytes[0] as char;
            if !matches!(prefix, 'M' | 'H' | 'D' | 'W') {
                return None;
            }
            if bytes[1..].iter().all(u8::is_ascii_digit) {
                Some(base.to_string())
            } else {
                None
            }
        })
        .unwrap_or(name_upper)
}

#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn scale_to_u32(value: f64, scale: f64) -> u32 {
    if !value.is_finite() {
        return 0;
    }
    let scaled = (value * scale).round();
    let clamped = scaled.clamp(0.0, f64::from(u32::MAX));
    clamped as u32
}

fn u64_to_usize(value: u64) -> Option<usize> {
    usize::try_from(value).ok()
}

fn u64_from_usize(value: usize) -> u64 {
    u64::try_from(value).unwrap_or(u64::MAX)
}

fn random_seed() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .ok()
        .and_then(|dur| u64::try_from(dur.as_nanos()).ok())
        .unwrap_or(42)
}

fn unit_value_per_price(spec: &omega_execution::SymbolSpec) -> f64 {
    if let (Some(tick_value), Some(tick_size)) = (spec.tick_value, spec.tick_size)
        && tick_size > 0.0
    {
        return tick_value / tick_size;
    }
    spec.contract_size.unwrap_or(1.0)
}

fn config_path(rel: &str) -> PathBuf {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest_dir.join("../../../").join(rel)
}

/// Resolves the execution costs file path.
///
/// Uses `OMEGA_EXECUTION_COSTS_FILE` env var if set, otherwise defaults to
/// `configs/execution_costs.yaml` relative to the repo root.
fn resolve_execution_costs_path() -> PathBuf {
    std::env::var("OMEGA_EXECUTION_COSTS_FILE")
        .ok()
        .map_or_else(
            || config_path("configs/execution_costs.yaml"),
            PathBuf::from,
        )
}

/// Resolves the symbol specs file path.
///
/// Uses `OMEGA_SYMBOL_SPECS_FILE` env var if set, otherwise defaults to
/// `configs/symbol_specs.yaml` relative to the repo root.
fn resolve_symbol_specs_path() -> PathBuf {
    std::env::var("OMEGA_SYMBOL_SPECS_FILE")
        .ok()
        .map_or_else(
            || config_path("configs/symbol_specs.yaml"),
            PathBuf::from,
        )
}

/// Date parsing boundary for date-only inputs.
#[derive(Clone, Copy)]
pub(crate) enum DateBoundary {
    /// Start of the day (00:00:00).
    Start,
    /// End of the day (23:59:59.999999999).
    End,
}

/// Parses an ISO-like datetime string to epoch nanoseconds.
pub(crate) fn parse_datetime_ns(value: &str, boundary: DateBoundary) -> Result<i64, BacktestError> {
    let value = value.trim();
    if value.is_empty() {
        return Err(BacktestError::DateParse("empty datetime".to_string()));
    }

    let (date_part, time_part) = if let Some((date, time)) = value.split_once('T') {
        (date, Some(time))
    } else if let Some((date, time)) = value.split_once(' ') {
        (date, Some(time))
    } else {
        (value, None)
    };

    let (year, month, day) = parse_date(date_part)?;
    let (hour, minute, second, nanos, offset) = if let Some(time) = time_part {
        parse_time(time)?
    } else {
        match boundary {
            DateBoundary::Start => (0, 0, 0, 0, 0),
            DateBoundary::End => (23, 59, 59, 999_999_999, 0),
        }
    };

    let days = days_from_civil(year, month, day);
    let mut total_seconds = days
        .checked_mul(86_400)
        .and_then(|v| v.checked_add(i64::from(hour) * 3_600))
        .and_then(|v| v.checked_add(i64::from(minute) * 60))
        .and_then(|v| v.checked_add(i64::from(second)))
        .ok_or_else(|| BacktestError::DateParse("datetime overflow".to_string()))?;

    total_seconds = total_seconds
        .checked_sub(i64::from(offset))
        .ok_or_else(|| BacktestError::DateParse("datetime overflow".to_string()))?;

    let total_ns = i128::from(total_seconds)
        .checked_mul(1_000_000_000)
        .and_then(|v| v.checked_add(i128::from(nanos)))
        .ok_or_else(|| BacktestError::DateParse("datetime overflow".to_string()))?;

    i64::try_from(total_ns).map_err(|_| BacktestError::DateParse("datetime overflow".to_string()))
}

fn parse_date(date: &str) -> Result<(i32, u32, u32), BacktestError> {
    let parts: Vec<&str> = date.split('-').collect();
    if parts.len() != 3 {
        return Err(BacktestError::DateParse(format!(
            "invalid date format: {date}"
        )));
    }
    let year: i32 = parts[0]
        .parse()
        .map_err(|_| BacktestError::DateParse(format!("invalid year: {date}")))?;
    let month: u32 = parts[1]
        .parse()
        .map_err(|_| BacktestError::DateParse(format!("invalid month: {date}")))?;
    let day: u32 = parts[2]
        .parse()
        .map_err(|_| BacktestError::DateParse(format!("invalid day: {date}")))?;

    if month == 0 || month > 12 {
        return Err(BacktestError::DateParse(format!("invalid month: {date}")));
    }
    let max_day = days_in_month(year, month);
    if day == 0 || day > max_day {
        return Err(BacktestError::DateParse(format!("invalid day: {date}")));
    }

    Ok((year, month, day))
}

fn parse_time(time: &str) -> Result<(u32, u32, u32, u32, i32), BacktestError> {
    let mut time_part = time.trim();
    let mut offset_secs: i32 = 0;

    if let Some(stripped) = time_part.strip_suffix('Z') {
        time_part = stripped;
    } else if let Some(idx) = time_part.rfind(['+', '-'])
        && idx > 0
    {
        let (t, offset) = time_part.split_at(idx);
        time_part = t;
        offset_secs = parse_offset(offset)?;
    }

    let parts: Vec<&str> = time_part.split(':').collect();
    if parts.len() < 2 || parts.len() > 3 {
        return Err(BacktestError::DateParse(format!(
            "invalid time format: {time}"
        )));
    }

    let hour: u32 = parts[0]
        .parse()
        .map_err(|_| BacktestError::DateParse(format!("invalid hour: {time}")))?;
    let minute: u32 = parts[1]
        .parse()
        .map_err(|_| BacktestError::DateParse(format!("invalid minute: {time}")))?;

    let (second, nanos) = if parts.len() == 3 {
        parse_seconds(parts[2])?
    } else {
        (0, 0)
    };

    if hour > 23 || minute > 59 || second > 59 {
        return Err(BacktestError::DateParse(format!(
            "invalid time value: {time}"
        )));
    }

    Ok((hour, minute, second, nanos, offset_secs))
}

fn parse_offset(offset: &str) -> Result<i32, BacktestError> {
    let sign = if offset.starts_with('+') {
        1
    } else if offset.starts_with('-') {
        -1
    } else {
        return Err(BacktestError::DateParse(format!(
            "invalid offset: {offset}"
        )));
    };

    let offset = &offset[1..];
    let parts: Vec<&str> = offset.split(':').collect();
    if parts.len() != 2 {
        return Err(BacktestError::DateParse(format!(
            "invalid offset: {offset}"
        )));
    }

    let hours: i32 = parts[0]
        .parse()
        .map_err(|_| BacktestError::DateParse(format!("invalid offset: {offset}")))?;
    let minutes: i32 = parts[1]
        .parse()
        .map_err(|_| BacktestError::DateParse(format!("invalid offset: {offset}")))?;

    if hours.abs() > 23 || minutes.abs() > 59 {
        return Err(BacktestError::DateParse(format!(
            "invalid offset: {offset}"
        )));
    }

    Ok(sign * (hours * 3_600 + minutes * 60))
}

fn parse_seconds(input: &str) -> Result<(u32, u32), BacktestError> {
    let mut parts = input.split('.');
    let seconds_str = parts.next().unwrap_or("");
    let seconds: u32 = seconds_str
        .parse()
        .map_err(|_| BacktestError::DateParse(format!("invalid seconds: {input}")))?;
    let nanos = if let Some(frac) = parts.next() {
        let mut digits = frac.trim().as_bytes().to_vec();
        if digits.len() > 9 {
            digits.truncate(9);
        }
        while digits.len() < 9 {
            digits.push(b'0');
        }
        let frac_str = std::str::from_utf8(&digits).unwrap_or("000000000");
        frac_str
            .parse::<u32>()
            .map_err(|_| BacktestError::DateParse(format!("invalid fraction: {input}")))?
    } else {
        0
    };

    Ok((seconds, nanos))
}

fn days_from_civil(year: i32, month: u32, day: u32) -> i64 {
    let y = i64::from(year) - i64::from(month <= 2);
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let m = i64::from(month);
    let d = i64::from(day);
    let doy = (153 * (m + if m > 2 { -3 } else { 9 }) + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

fn days_in_month(year: i32, month: u32) -> u32 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 => {
            if is_leap_year(year) {
                29
            } else {
                28
            }
        }
        _ => 0,
    }
}

fn is_leap_year(year: i32) -> bool {
    (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0)
}

struct SlippageMultiplier {
    inner: Box<dyn SlippageModel>,
    multiplier: f64,
}

impl SlippageMultiplier {
    fn new(inner: Box<dyn SlippageModel>, multiplier: f64) -> Self {
        Self {
            inner,
            multiplier,
        }
    }
}

impl SlippageModel for SlippageMultiplier {
    fn calculate(
        &self,
        price: f64,
        direction: Direction,
        rng: &mut rand_chacha::ChaCha8Rng,
    ) -> f64 {
        self.inner.calculate(price, direction, rng) * self.multiplier
    }

    fn name(&self) -> &'static str {
        "SlippageMultiplier"
    }
}

struct FeeMultiplier {
    inner: Box<dyn FeeModel>,
    multiplier: f64,
}

impl FeeMultiplier {
    fn new(inner: Box<dyn FeeModel>, multiplier: f64) -> Self {
        Self { inner, multiplier }
    }
}

impl FeeModel for FeeMultiplier {
    fn calculate(&self, size: f64, price: f64) -> f64 {
        self.inner.calculate(size, price) * self.multiplier
    }

    fn name(&self) -> &'static str {
        "FeeMultiplier"
    }
}

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::path::Path;
    use std::sync::Arc;

    use arrow::array::{ArrayRef, Float64Array, TimestampNanosecondArray};
    use arrow::datatypes::{DataType, Field, Schema, TimeUnit};
    use arrow::record_batch::RecordBatch;
    use omega_types::{ExitReason, OrderType, Signal};
    use parquet::arrow::arrow_writer::ArrowWriter;
    use temp_env::with_var;
    use tempfile::tempdir;

    use super::*;

    #[test]
    fn test_parse_date_only_end_of_day() {
        let start = parse_datetime_ns("2024-01-01", DateBoundary::Start).unwrap();
        let end = parse_datetime_ns("2024-01-01", DateBoundary::End).unwrap();
        assert!(end > start);
        assert_eq!(end - start, 86_399_000_000_000 + 999_999_999);
    }

    #[test]
    fn test_parse_datetime_with_offset() {
        let ts = parse_datetime_ns("2024-01-01T01:00:00+01:00", DateBoundary::Start).unwrap();
        let ts_utc = parse_datetime_ns("2024-01-01T00:00:00Z", DateBoundary::Start).unwrap();
        assert_eq!(ts, ts_utc);
    }

    const STEP_NS: i64 = 60_000_000_000;
    const BASE_TS: i64 = 1_704_067_200_000_000_000;

    fn make_candles(closes: &[f64], high_delta: f64, low_delta: f64) -> Vec<Candle> {
        closes
            .iter()
            .enumerate()
            .map(|(idx, close)| {
                let idx_i64 = i64::try_from(idx).expect("idx fits in i64");
                let ts = BASE_TS + idx_i64 * STEP_NS;
                Candle {
                    timestamp_ns: ts,
                    close_time_ns: ts + STEP_NS - 1,
                    open: *close,
                    high: *close + high_delta,
                    low: *close - low_delta,
                    close: *close,
                    volume: 100.0,
                }
            })
            .collect()
    }

    fn write_custom_parquet(
        path: &Path,
        fields: Vec<Field>,
        columns: Vec<ArrayRef>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let schema = Arc::new(Schema::new(fields));
        let batch = RecordBatch::try_new(schema.clone(), columns)?;
        let file = File::create(path)?;
        let mut writer = ArrowWriter::try_new(file, schema, None)?;
        writer.write(&batch)?;
        writer.close().map(|_| ()).map_err(Into::into)
    }

    fn write_candle_parquet(
        path: &Path,
        candles: &[Candle],
    ) -> Result<(), Box<dyn std::error::Error>> {
        let timestamps: Vec<i64> = candles.iter().map(|c| c.timestamp_ns).collect();
        let opens: Vec<f64> = candles.iter().map(|c| c.open).collect();
        let highs: Vec<f64> = candles.iter().map(|c| c.high).collect();
        let lows: Vec<f64> = candles.iter().map(|c| c.low).collect();
        let closes: Vec<f64> = candles.iter().map(|c| c.close).collect();
        let volumes: Vec<f64> = candles.iter().map(|c| c.volume).collect();

        let fields = vec![
            Field::new(
                "UTC time",
                DataType::Timestamp(TimeUnit::Nanosecond, Some("UTC".into())),
                false,
            ),
            Field::new("Open", DataType::Float64, false),
            Field::new("High", DataType::Float64, false),
            Field::new("Low", DataType::Float64, false),
            Field::new("Close", DataType::Float64, false),
            Field::new("Volume", DataType::Float64, false),
        ];

        let columns: Vec<ArrayRef> = vec![
            Arc::new(TimestampNanosecondArray::from(timestamps).with_timezone("UTC")),
            Arc::new(Float64Array::from(opens)),
            Arc::new(Float64Array::from(highs)),
            Arc::new(Float64Array::from(lows)),
            Arc::new(Float64Array::from(closes)),
            Arc::new(Float64Array::from(volumes)),
        ];

        write_custom_parquet(path, fields, columns)
    }

    fn setup_engine_with_fixture() -> BacktestEngine {
        let dir = tempdir().expect("tempdir");
        let root = dir.path();
        let symbol_dir = root.join("EURUSD");
        std::fs::create_dir_all(&symbol_dir).expect("symbol dir");

        let bid = make_candles(&[1.0, 1.1, 0.9], 0.02, 0.02);
        let ask: Vec<Candle> = bid
            .iter()
            .map(|c| Candle {
                timestamp_ns: c.timestamp_ns,
                close_time_ns: c.close_time_ns,
                open: c.open + 0.0002,
                high: c.high + 0.0002,
                low: c.low + 0.0002,
                close: c.close + 0.0002,
                volume: c.volume,
            })
            .collect();

        let bid_path = symbol_dir.join("EURUSD_M1_BID.parquet");
        let ask_path = symbol_dir.join("EURUSD_M1_ASK.parquet");
        write_candle_parquet(&bid_path, &bid).expect("bid parquet");
        write_candle_parquet(&ask_path, &ask).expect("ask parquet");

        let config = serde_json::json!({
            "schema_version": "2",
            "strategy_name": "mean_reversion_z_score",
            "symbol": "EURUSD",
            "start_date": "2024-01-01T00:00:00Z",
            "end_date": "2024-01-01T00:02:00Z",
            "run_mode": "dev",
            "data_mode": "candle",
            "execution_variant": "v2",
            "timeframes": {
                "primary": "M1",
                "additional": [],
                "additional_source": "separate_parquet"
            },
            "warmup_bars": 2,
            "rng_seed": 42,
            "costs": {"enabled": false},
            "strategy_parameters": {
                "ema_length": 2,
                "atr_length": 1,
                "atr_mult": 1.0,
                "window_length": 2,
                "z_score_long": -0.5,
                "z_score_short": 0.5,
                "htf_filter": "none",
                "extra_htf_filter": "none",
                "enabled_scenarios": []
            }
        });

        let config_json = serde_json::to_string(&config).expect("config json");

        with_var("OMEGA_DATA_PARQUET_ROOT", Some(root), || {
            let cfg: BacktestConfig = serde_json::from_str(&config_json).expect("config parse");
            BacktestEngine::new(cfg).expect("engine")
        })
    }

    #[test]
    fn test_pending_trigger_before_stops_allows_same_bar_exit() {
        let mut engine = setup_engine_with_fixture();

        let signal = Signal {
            direction: Direction::Long,
            order_type: OrderType::Limit,
            entry_price: 1.0,
            stop_loss: 0.99,
            take_profit: 1.02,
            size: Some(1.0),
            scenario_id: 1,
            tags: Vec::new(),
            meta: serde_json::json!({}),
        };

        let created_at = BASE_TS + STEP_NS;
        engine
            .execution
            .add_pending_order(&signal, created_at, &engine.symbol_costs)
            .expect("pending order");

        engine.process_bar(2);

        let trades = engine.portfolio.closed_trades();
        assert_eq!(trades.len(), 1);
        assert_eq!(trades[0].exit_time_ns, BASE_TS + STEP_NS * 2);
        assert_eq!(trades[0].reason, ExitReason::StopLoss);
    }
}
