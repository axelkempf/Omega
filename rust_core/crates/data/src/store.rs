use omega_types::{Candle, Timeframe};

/// Hauptdatenstruktur für aligned Bid/Ask Candles
#[derive(Debug, Clone)]
pub struct CandleStore {
    pub bid: Vec<Candle>,
    pub ask: Vec<Candle>,
    pub timestamps: Vec<i64>,
    pub timeframe: Timeframe,
    pub symbol: String,
    pub warmup_bars: usize,
}

impl CandleStore {
    pub fn len(&self) -> usize {
        self.bid.len()
    }

    pub fn is_empty(&self) -> bool {
        self.bid.is_empty()
    }

    pub fn trading_bars(&self) -> usize {
        self.len().saturating_sub(self.warmup_bars)
    }

    /// Gibt (bid, ask) für Index zurück
    pub fn get(&self, idx: usize) -> Option<(&Candle, &Candle)> {
        if idx < self.len() {
            Some((&self.bid[idx], &self.ask[idx]))
        } else {
            None
        }
    }
}

/// Multi-Timeframe Store für Primary + HTF
#[derive(Debug, Clone)]
pub struct MultiTfStore {
    pub primary: CandleStore,
    pub htf: Option<CandleStore>,
    /// Mapping: Primary-Index → HTF-Index (letzte abgeschlossene HTF-Bar)
    pub htf_index_map: Vec<Option<usize>>,
}
