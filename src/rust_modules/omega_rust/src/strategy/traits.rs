//! RustStrategy Trait Definition
//!
//! Der zentrale Trait, den alle Rust-Strategien implementieren müssen.

use crate::indicators::IndicatorCache;
use super::types::{DataSlice, Position, PositionAction, StrategyConfig, Timeframe, TradeSignal};

/// Fehlertyp für Strategie-Operationen
#[derive(Debug, Clone)]
pub enum StrategyError {
    /// Konfigurationsfehler
    ConfigError(String),
    /// Ungültiger Parameter
    InvalidParameter { name: String, reason: String },
    /// Nicht genug Daten für Warmup
    InsufficientData { required: usize, available: usize },
    /// Indikator-Fehler
    IndicatorError(String),
    /// Interner Fehler
    InternalError(String),
}

impl std::fmt::Display for StrategyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StrategyError::ConfigError(msg) => write!(f, "Config error: {}", msg),
            StrategyError::InvalidParameter { name, reason } => {
                write!(f, "Invalid parameter '{}': {}", name, reason)
            }
            StrategyError::InsufficientData { required, available } => {
                write!(
                    f,
                    "Insufficient data: required {} bars, got {}",
                    required, available
                )
            }
            StrategyError::IndicatorError(msg) => write!(f, "Indicator error: {}", msg),
            StrategyError::InternalError(msg) => write!(f, "Internal error: {}", msg),
        }
    }
}

impl std::error::Error for StrategyError {}

/// Core Trait für alle Rust-Strategien
///
/// Strategien implementieren diesen Trait um im Pure-Rust-Backtest ausgeführt zu werden.
/// Das Ziel ist die Eliminierung des FFI-Overheads durch vollständige Ausführung in Rust.
///
/// # Performance-Anforderungen
///
/// - `evaluate()`: ≤1μs pro Bar
/// - `manage_position()`: ≤500ns pro Aufruf
/// - `on_init()`: ≤100ms einmalig
///
/// # Thread Safety
///
/// Strategien müssen `Send + Sync` sein für parallelisierte Backtests.
///
/// # Beispiel
///
/// ```rust,ignore
/// struct MyStrategy {
///     lookback: usize,
///     threshold: f64,
/// }
///
/// impl RustStrategy for MyStrategy {
///     fn evaluate(&self, slice: &DataSlice, cache: &mut IndicatorCache) -> Option<TradeSignal> {
///         let zscore = slice.indicator("zscore_100")?;
///         if zscore < -self.threshold {
///             let bid = slice.current_bid()?;
///             Some(TradeSignal::long(
///                 bid.close,
///                 bid.close - 0.0050,
///                 bid.close + 0.0100,
///                 slice.symbol.clone(),
///                 1,
///                 slice.timestamp_us,
///             ))
///         } else {
///             None
///         }
///     }
///     // ... weitere Methoden
/// }
/// ```
pub trait RustStrategy: Send + Sync {
    /// Evaluiert einen einzelnen Bar und generiert optional ein Trade-Signal.
    ///
    /// Diese Methode wird für **jeden Bar** im Backtest aufgerufen (nach Warmup).
    /// Die Implementierung muss side-effect-frei sein (pure function).
    ///
    /// # Arguments
    ///
    /// * `slice` - Snapshot der aktuellen Marktdaten
    /// * `cache` - IndicatorCache mit pre-computed Indikatoren
    ///
    /// # Returns
    ///
    /// * `Some(TradeSignal)` - Neuer Trade soll geöffnet werden
    /// * `None` - Kein Trade
    ///
    /// # Performance
    ///
    /// Target: ≤1μs pro Aufruf. Keine Heap-Allokation in Hot-Path.
    fn evaluate(&self, slice: &DataSlice, cache: &mut IndicatorCache) -> Option<TradeSignal>;

    /// Verwaltet eine offene Position.
    ///
    /// Wird für jede offene Position pro Bar aufgerufen. Ermöglicht:
    /// - Trailing Stop
    /// - Break-Even
    /// - Timeout-basierte Schließung
    ///
    /// # Arguments
    ///
    /// * `position` - Die zu verwaltende Position
    /// * `slice` - Aktuelle Marktdaten
    ///
    /// # Returns
    ///
    /// * `PositionAction::Hold` - Keine Änderung
    /// * `PositionAction::ModifyStopLoss` - SL anpassen
    /// * `PositionAction::Close` - Position schließen
    fn manage_position(&self, position: &Position, slice: &DataSlice) -> PositionAction;

    /// Gibt den primären Timeframe der Strategie zurück.
    ///
    /// Dieser Timeframe bestimmt die Hauptfrequenz der `evaluate()`-Aufrufe.
    fn primary_timeframe(&self) -> Timeframe;

    /// Minimale Anzahl Bars für Warmup.
    ///
    /// Die ersten `warmup_bars()` werden nicht für Trading evaluiert,
    /// sondern nur für Indikator-Berechnung verwendet.
    fn warmup_bars(&self) -> usize {
        200 // Default: 200 Bars
    }

    /// Maximale Anzahl gleichzeitig offener Positionen.
    fn max_positions(&self) -> usize {
        1 // Default: 1 Position gleichzeitig
    }

    /// Initialisierung nach Config-Load.
    ///
    /// Wird einmalig vor dem Backtest aufgerufen.
    fn on_init(&mut self, _config: &StrategyConfig) -> Result<(), StrategyError> {
        Ok(())
    }

    /// Cleanup bei Strategy-Deallokation.
    fn on_deinit(&mut self) {
        // Default: no-op
    }

    /// Name der Strategie (für Logging und Registry).
    fn name(&self) -> &str;

    /// Version der Strategie.
    fn version(&self) -> &str {
        "1.0.0"
    }

    /// Beschreibung der Strategie.
    fn description(&self) -> &str {
        ""
    }

    /// Erlaubte Richtungen: "long", "short", "both"
    fn direction_filter(&self) -> &str {
        "both"
    }

    /// Aktivierte Szenarien (1-6)
    fn enabled_scenarios(&self) -> &[u8] {
        &[1, 2, 3, 4, 5, 6]
    }

    /// Prüft ob Szenario aktiviert ist.
    fn is_scenario_enabled(&self, scenario: u8) -> bool {
        self.enabled_scenarios().contains(&scenario)
    }

    /// Prüft ob Richtung erlaubt ist.
    fn is_direction_allowed(&self, direction: &str) -> bool {
        let filter = self.direction_filter();
        filter == "both" || filter == direction
    }
}

/// Factory-Trait für Strategie-Erstellung
pub trait StrategyFactory: Send + Sync {
    /// Erstellt eine neue Strategie-Instanz aus Konfiguration.
    fn create(&self, config: &StrategyConfig) -> Result<Box<dyn RustStrategy>, StrategyError>;

    /// Name der Factory (entspricht Registry-Key).
    fn name(&self) -> &str;
}

#[cfg(test)]
mod tests {
    use super::*;

    struct DummyStrategy;

    impl RustStrategy for DummyStrategy {
        fn evaluate(&self, _slice: &DataSlice, _cache: &mut IndicatorCache) -> Option<TradeSignal> {
            None
        }

        fn manage_position(&self, _position: &Position, _slice: &DataSlice) -> PositionAction {
            PositionAction::Hold()
        }

        fn primary_timeframe(&self) -> Timeframe {
            Timeframe::M5
        }

        fn name(&self) -> &str {
            "dummy"
        }
    }

    #[test]
    fn test_default_values() {
        let strategy = DummyStrategy;
        assert_eq!(strategy.warmup_bars(), 200);
        assert_eq!(strategy.max_positions(), 1);
        assert_eq!(strategy.version(), "1.0.0");
        assert_eq!(strategy.direction_filter(), "both");
        assert!(strategy.is_direction_allowed("long"));
        assert!(strategy.is_direction_allowed("short"));
        assert!(strategy.is_scenario_enabled(1));
        assert!(strategy.is_scenario_enabled(6));
    }
}
