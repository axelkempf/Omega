//! Strategy runner utilities for building `BarContext` with multi-TF support.

use std::cell::RefCell;
use std::collections::HashSet;

use omega_data::MultiTfStore;
use omega_indicators::{IndicatorCache, MultiTfIndicatorCache};
use omega_types::Signal;

use crate::context::{BarContext, HtfContext};
use crate::traits::Strategy;

/// Builder for `BarContext` that injects `MultiTfIndicatorCache` and HTF context.
pub struct BarContextBuilder<'a> {
    store: &'a MultiTfStore,
    indicators: &'a IndicatorCache,
    htf_indicators: Option<&'a IndicatorCache>,
    multi_tf: RefCell<MultiTfIndicatorCache>,
    empty_htf_cache: IndicatorCache,
}

impl<'a> BarContextBuilder<'a> {
    /// Creates a new `BarContextBuilder` with a multi-TF cache.
    #[must_use]
    pub fn new(
        store: &'a MultiTfStore,
        indicators: &'a IndicatorCache,
        htf_indicators: Option<&'a IndicatorCache>,
    ) -> Self {
        let multi_tf = RefCell::new(build_multi_tf_cache(store));

        Self {
            store,
            indicators,
            htf_indicators,
            multi_tf,
            empty_htf_cache: IndicatorCache::new(),
        }
    }

    /// Returns primary series length.
    pub fn primary_len(&self) -> usize {
        self.store.primary.len()
    }

    /// Builds a `BarContext` (plus optional HTF context) for the given index.
    pub fn context_at(
        &'a self,
        idx: usize,
        session_open: bool,
        news_blocked: bool,
    ) -> Option<BarContext<'a>> {
        let (bid, ask) = self.store.primary.get(idx)?;
        let mut ctx = BarContext::new(idx, bid.timestamp_ns, bid, ask, self.indicators)
            .with_multi_tf(&self.multi_tf)
            .with_session(session_open)
            .with_news_blocked(news_blocked);

        if let Some(htf_ctx) = self.build_htf_context(idx) {
            ctx = ctx.with_htf(htf_ctx);
        }

        Some(ctx)
    }

    fn build_htf_context(&'a self, idx: usize) -> Option<HtfContext<'a>> {
        let htf_store = self.store.htf.as_ref()?;
        let htf_idx = self.store.htf_index_at(idx)?;
        let (htf_bid, htf_ask) = htf_store.get(htf_idx)?;
        let cache = self.htf_indicators.unwrap_or(&self.empty_htf_cache);
        Some(HtfContext::new(
            htf_bid,
            htf_ask,
            cache,
            htf_idx,
            htf_store.timeframe.as_str(),
        ))
    }
}

/// Minimal strategy runner that builds `BarContext` with multi-TF cache.
pub struct StrategyRunner<'a, S: Strategy> {
    strategy: &'a mut S,
    builder: BarContextBuilder<'a>,
}

impl<'a, S: Strategy> StrategyRunner<'a, S> {
    /// Creates a new runner for the given strategy and data store.
    pub fn new(
        strategy: &'a mut S,
        store: &'a MultiTfStore,
        indicators: &'a IndicatorCache,
        htf_indicators: Option<&'a IndicatorCache>,
    ) -> Self {
        let builder = BarContextBuilder::new(store, indicators, htf_indicators);
        Self { strategy, builder }
    }

    /// Runs the strategy over all primary bars with session/news gates.
    pub fn run_with<F, G>(&mut self, session_open: F, news_blocked: G) -> Vec<Option<Signal>>
    where
        F: Fn(usize) -> bool,
        G: Fn(usize) -> bool,
    {
        let mut signals = Vec::with_capacity(self.builder.primary_len());
        for idx in 0..self.builder.primary_len() {
            let open = session_open(idx);
            let blocked = news_blocked(idx);
            let signal = self
                .builder
                .context_at(idx, open, blocked)
                .and_then(|ctx| self.strategy.on_bar(&ctx));
            signals.push(signal);
        }
        signals
    }

    /// Runs the strategy with default gates (session open, news not blocked).
    pub fn run(&mut self) -> Vec<Option<Signal>> {
        self.run_with(|_| true, |_| false)
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use omega_data::CandleStore;
    use omega_types::Candle;
    use omega_types::Timeframe;

    fn make_store(timeframe: Timeframe, closes: &[f64], step: i64) -> CandleStore {
        let mut bid = Vec::new();
        let mut ask = Vec::new();
        let mut timestamps = Vec::new();
        for (i, close) in closes.iter().enumerate() {
            let idx = i64::try_from(i).unwrap_or(i64::MAX / step.max(1));
            let ts = idx * step;
            let close_time_ns = ts + step - 1;
            bid.push(make_candle(ts, close_time_ns, *close));
            ask.push(make_candle(ts, close_time_ns, *close + 0.0002));
            timestamps.push(ts);
        }
        CandleStore {
            bid,
            ask,
            timestamps,
            timeframe,
            symbol: "EURUSD".to_string(),
            warmup_bars: 0,
        }
    }

    fn make_candle(timestamp_ns: i64, close_time_ns: i64, close: f64) -> Candle {
        Candle {
            timestamp_ns,
            close_time_ns,
            open: close,
            high: close,
            low: close,
            close,
            volume: 0.0,
        }
    }

    #[test]
    fn test_context_builder_injects_multi_tf_and_htf() {
        let primary = make_store(Timeframe::M1, &[1.0, 1.01, 1.02, 1.03], 60);
        let htf = make_store(Timeframe::H1, &[1.0, 1.05], 3600);
        let store = MultiTfStore::new(primary, Some(htf), Vec::new());

        let indicators = IndicatorCache::new();
        let builder = BarContextBuilder::new(&store, &indicators, None);

        let ctx = builder
            .context_at(1, true, false)
            .expect("expected context");

        assert!(ctx.multi_tf.is_some());
        assert!(ctx.htf_data.is_some());
        assert_eq!(ctx.htf_data.as_ref().unwrap().timeframe, "H1");
    }
}
