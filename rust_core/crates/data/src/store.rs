//! Candle stores and multi-timeframe mapping.

use omega_types::{Candle, Timeframe};

/// Hauptdatenstruktur für aligned Bid/Ask Candles.
#[derive(Debug, Clone)]
pub struct CandleStore {
    /// Aligned bid candles.
    pub bid: Vec<Candle>,
    /// Aligned ask candles.
    pub ask: Vec<Candle>,
    /// Aligned timestamps in UTC epoch-ns.
    pub timestamps: Vec<i64>,
    /// Timeframe of the candles.
    pub timeframe: Timeframe,
    /// Symbol identifier.
    pub symbol: String,
    /// Configured warmup bars for this store.
    pub warmup_bars: usize,
}

impl CandleStore {
    /// Returns the number of aligned bars.
    #[must_use]
    pub fn len(&self) -> usize {
        self.bid.len()
    }

    /// Returns `true` if there are no bars.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.bid.is_empty()
    }

    /// Returns the number of trading bars after warmup.
    #[must_use]
    pub fn trading_bars(&self) -> usize {
        self.len().saturating_sub(self.warmup_bars)
    }

    /// Gibt (bid, ask) für Index zurück.
    #[must_use]
    pub fn get(&self, idx: usize) -> Option<(&Candle, &Candle)> {
        if idx < self.len() {
            Some((&self.bid[idx], &self.ask[idx]))
        } else {
            None
        }
    }
}

/// Multi-Timeframe Store für Primary + HTF.
#[derive(Debug, Clone)]
pub struct MultiTfStore {
    /// Primary timeframe store.
    pub primary: CandleStore,
    /// Optional HTF store.
    pub htf: Option<CandleStore>,
    /// Additional timeframes (e.g. scenario6 overlays).
    pub additional: Vec<CandleStore>,
    /// Mapping: Primary-Index → HTF-Index (letzte abgeschlossene HTF-Bar).
    pub htf_index_map: Vec<Option<usize>>,
}

impl MultiTfStore {
    /// Creates a new `MultiTfStore` with a computed HTF index map.
    #[must_use]
    pub fn new(
        primary: CandleStore,
        htf: Option<CandleStore>,
        additional: Vec<CandleStore>,
    ) -> Self {
        let htf_index_map = match &htf {
            Some(htf_store) => build_index_map(&primary.timestamps, &htf_store.timestamps),
            None => vec![None; primary.timestamps.len()],
        };

        Self {
            primary,
            htf,
            additional,
            htf_index_map,
        }
    }

    /// Returns the mapped HTF index for a primary bar index.
    #[must_use]
    pub fn htf_index_at(&self, idx: usize) -> Option<usize> {
        self.htf_index_map.get(idx).and_then(|v| *v)
    }
}

fn build_index_map(primary_ts: &[i64], htf_ts: &[i64]) -> Vec<Option<usize>> {
    if primary_ts.is_empty() {
        return Vec::new();
    }

    if htf_ts.is_empty() {
        return vec![None; primary_ts.len()];
    }

    let mut mapping = Vec::with_capacity(primary_ts.len());
    let mut j = 0usize;
    let mut last_valid: Option<usize> = None;

    for &ts in primary_ts {
        while j < htf_ts.len() && htf_ts[j] <= ts {
            last_valid = Some(j);
            j += 1;
        }
        mapping.push(last_valid);
    }

    mapping
}
