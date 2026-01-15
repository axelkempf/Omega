use omega_types::Candle;
use proptest::prelude::*;

/// Generiert valide Candle-Sequenzen für Property-Tests
pub fn valid_candle_sequence(len: usize) -> impl Strategy<Value = Vec<Candle>> {
    prop::collection::vec(valid_candle(), len..=len).prop_map(|mut candles| {
        // Sortiere nach Timestamp und mache monoton
        candles.sort_by_key(|c| c.timestamp_ns);
        let mut ts = 1_704_067_200_000_000_000i64; // 2024-01-01 00:00:00 UTC
        for candle in &mut candles {
            candle.timestamp_ns = ts;
            ts += 60_000_000_000; // +1 Minute
        }
        candles
    })
}

fn valid_candle() -> impl Strategy<Value = Candle> {
    (
        1.0f64..2.0,  // base price (z.B. EURUSD range)
        0.0001..0.01, // spread range
    )
        .prop_map(|(base, spread)| {
            let low = base - spread;
            let high = base + spread;
            Candle {
                timestamp_ns: 0, // wird in valid_candle_sequence überschrieben
                open: base,
                high,
                low,
                close: base + (spread * 0.5),
                volume: 100.0,
            }
        })
}
