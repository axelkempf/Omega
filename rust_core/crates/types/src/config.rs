/// Main backtest configuration
#[derive(Debug, Clone, serde::Serialize)]
pub struct BacktestConfig {
    /// Schema version
    pub schema_version: String,
    /// Strategy name
    pub strategy_name: String,
    /// Trading symbol
    pub symbol: String,
    /// Start date (ISO format)
    pub start_date: String,
    /// End date (ISO format)
    pub end_date: String,
    /// Run mode
    pub run_mode: RunMode,
    /// Data mode
    pub data_mode: DataMode,
    /// Execution variant
    pub execution_variant: ExecutionVariant,
    /// Timeframe configuration
    pub timeframes: TimeframeConfig,
    /// Warmup bars count
    #[serde(default = "default_warmup")]
    pub warmup_bars: usize,
    /// RNG seed for reproducibility
    #[serde(default)]
    pub rng_seed: Option<u64>,
    /// Trading session windows
    #[serde(default)]
    pub sessions: Option<Vec<SessionConfig>>,
    /// Account configuration
    #[serde(default)]
    pub account: AccountConfig,
    /// Costs configuration
    #[serde(default)]
    pub costs: CostsConfig,
    /// News filter configuration
    #[serde(default)]
    pub news_filter: Option<NewsFilterConfig>,
    /// Logging configuration
    #[serde(default)]
    pub logging: LoggingConfig,
    /// Metrics configuration
    #[serde(default)]
    pub metrics: Option<serde_json::Value>,
    /// Trade management configuration
    #[serde(default)]
    pub trade_management: Option<TradeManagementConfig>,
    /// Strategy-specific parameters
    #[serde(default)]
    pub strategy_parameters: serde_json::Value,
}

const DEFAULT_RNG_SEED: u64 = 42;

#[derive(Debug, Clone, serde::Deserialize)]
struct BacktestConfigRaw {
    pub schema_version: String,
    pub strategy_name: String,
    pub symbol: String,
    pub start_date: String,
    pub end_date: String,
    pub run_mode: RunMode,
    pub data_mode: DataMode,
    pub execution_variant: ExecutionVariant,
    pub timeframes: TimeframeConfig,
    #[serde(default = "default_warmup")]
    pub warmup_bars: usize,
    #[serde(default)]
    pub rng_seed: Option<u64>,
    #[serde(default)]
    pub sessions: Option<Vec<SessionConfig>>,
    #[serde(default)]
    pub account: AccountConfig,
    #[serde(default)]
    pub costs: CostsConfig,
    #[serde(default)]
    pub news_filter: Option<NewsFilterConfig>,
    #[serde(default)]
    pub logging: LoggingConfig,
    #[serde(default)]
    pub metrics: Option<serde_json::Value>,
    #[serde(default)]
    pub trade_management: Option<TradeManagementConfig>,
    #[serde(default)]
    pub strategy_parameters: serde_json::Value,
}

impl From<BacktestConfigRaw> for BacktestConfig {
    fn from(raw: BacktestConfigRaw) -> Self {
        let rng_seed = match raw.run_mode {
            RunMode::Dev => Some(raw.rng_seed.unwrap_or(DEFAULT_RNG_SEED)),
            RunMode::Prod => raw.rng_seed,
        };

        Self {
            schema_version: raw.schema_version,
            strategy_name: raw.strategy_name,
            symbol: raw.symbol,
            start_date: raw.start_date,
            end_date: raw.end_date,
            run_mode: raw.run_mode,
            data_mode: raw.data_mode,
            execution_variant: raw.execution_variant,
            timeframes: raw.timeframes,
            warmup_bars: raw.warmup_bars,
            rng_seed,
            sessions: raw.sessions,
            account: raw.account,
            costs: raw.costs,
            news_filter: raw.news_filter,
            logging: raw.logging,
            metrics: raw.metrics,
            trade_management: raw.trade_management,
            strategy_parameters: raw.strategy_parameters,
        }
    }
}

impl<'de> serde::Deserialize<'de> for BacktestConfig {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let raw = BacktestConfigRaw::deserialize(deserializer)?;
        Ok(raw.into())
    }
}

fn default_warmup() -> usize {
    500
}

/// Run mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunMode {
    /// Development mode
    #[default]
    Dev,
    /// Production mode
    Prod,
}

/// Data mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DataMode {
    /// Candle data
    #[default]
    Candle,
    /// Tick data
    Tick,
}

/// Execution variant
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionVariant {
    /// V2 execution
    #[default]
    V2,
    /// V1 parity mode
    V1Parity,
}

/// Timeframe configuration
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TimeframeConfig {
    /// Primary timeframe
    pub primary: String,
    /// Additional timeframes
    #[serde(default)]
    pub additional: Vec<String>,
    /// Source for additional timeframes
    #[serde(default = "default_additional_source")]
    pub additional_source: String,
}

fn default_additional_source() -> String {
    "separate_parquet".to_string()
}

// ============================================
// SUB-CONFIGS
// ============================================

/// Session-Fenster für Trading (UTC)
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SessionConfig {
    /// Start-Zeit (HH:MM, UTC)
    pub start: String,
    /// End-Zeit (HH:MM, UTC) - wenn end <= start, gilt als cross-midnight
    pub end: String,
}

impl SessionConfig {
    /// Prüft ob ein Zeitpunkt (Sekunden seit Mitternacht UTC) in dieser Session liegt
    #[must_use]
    pub fn contains(&self, seconds_of_day: u32) -> bool {
        let start_secs = parse_hhmm_to_seconds(&self.start);
        let end_secs = parse_hhmm_to_seconds(&self.end);

        if end_secs > start_secs {
            // Normal: 08:00 - 17:00
            seconds_of_day >= start_secs && seconds_of_day < end_secs
        } else {
            // Cross-midnight: 22:00 - 06:00
            seconds_of_day >= start_secs || seconds_of_day < end_secs
        }
    }
}

fn parse_hhmm_to_seconds(hhmm: &str) -> u32 {
    let parts: Vec<&str> = hhmm.split(':').collect();
    let hours: u32 = parts[0].parse().unwrap_or(0);
    let minutes: u32 = parts.get(1).and_then(|m| m.parse().ok()).unwrap_or(0);
    hours * 3600 + minutes * 60
}

/// Account-/Sizing-Konfiguration
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AccountConfig {
    /// Initial account balance
    #[serde(default = "default_initial_balance")]
    pub initial_balance: f64,
    /// Account currency
    #[serde(default = "default_account_currency")]
    pub account_currency: String,
    /// Risk per trade
    #[serde(default = "default_risk_per_trade")]
    pub risk_per_trade: f64,
    /// Maximum open positions
    #[serde(default = "default_max_positions")]
    pub max_positions: usize,
}

fn default_initial_balance() -> f64 {
    10000.0
}
fn default_account_currency() -> String {
    "EUR".to_string()
}
fn default_risk_per_trade() -> f64 {
    100.0
}
fn default_max_positions() -> usize {
    1
}

impl Default for AccountConfig {
    fn default() -> Self {
        Self {
            initial_balance: default_initial_balance(),
            account_currency: default_account_currency(),
            risk_per_trade: default_risk_per_trade(),
            max_positions: default_max_positions(),
        }
    }
}

/// Kostenmodell-Konfiguration (Toggles + Multipliers)
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CostsConfig {
    /// Enable costs calculation
    #[serde(default = "default_costs_enabled")]
    pub enabled: bool,
    /// Fee multiplier
    #[serde(default = "default_multiplier")]
    pub fee_multiplier: f64,
    /// Slippage multiplier
    #[serde(default = "default_multiplier")]
    pub slippage_multiplier: f64,
    /// Spread multiplier
    #[serde(default = "default_multiplier")]
    pub spread_multiplier: f64,
    /// Symbol-spezifische `pip_size` (falls nicht aus `symbol_specs.yaml`)
    #[serde(default)]
    pub pip_size: Option<f64>,
    /// `pip_buffer_factor` für SL/TP Checks
    #[serde(default = "default_pip_buffer_factor")]
    pub pip_buffer_factor: f64,
}

fn default_costs_enabled() -> bool {
    true
}
fn default_multiplier() -> f64 {
    1.0
}
fn default_pip_buffer_factor() -> f64 {
    0.5
}

impl Default for CostsConfig {
    fn default() -> Self {
        Self {
            enabled: default_costs_enabled(),
            fee_multiplier: default_multiplier(),
            slippage_multiplier: default_multiplier(),
            spread_multiplier: default_multiplier(),
            pip_size: None,
            pip_buffer_factor: default_pip_buffer_factor(),
        }
    }
}

/// News impact level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum NewsImpact {
    /// Low impact
    Low,
    /// Medium impact
    #[default]
    Medium,
    /// High impact
    High,
}

/// News-Filter-Konfiguration
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct NewsFilterConfig {
    /// Enable news filter
    #[serde(default)]
    pub enabled: bool,
    /// Minutes before news event
    #[serde(default = "default_news_minutes")]
    pub minutes_before: u32,
    /// Minutes after news event
    #[serde(default = "default_news_minutes")]
    pub minutes_after: u32,
    /// Minimum impact level to filter
    #[serde(default = "default_min_impact")]
    pub min_impact: NewsImpact,
    /// Currencies to filter - wenn None, aus symbol abgeleitet
    #[serde(default)]
    pub currencies: Option<Vec<String>>,
}

fn default_news_minutes() -> u32 {
    30
}
fn default_min_impact() -> NewsImpact {
    NewsImpact::Medium
}

impl Default for NewsFilterConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            minutes_before: default_news_minutes(),
            minutes_after: default_news_minutes(),
            min_impact: default_min_impact(),
            currencies: None,
        }
    }
}

/// Logging-Konfiguration
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LoggingConfig {
    /// Enable entry logging
    #[serde(default)]
    pub enable_entry_logging: bool,
    /// Logging mode
    #[serde(default = "default_logging_mode")]
    pub logging_mode: String,
}

fn default_logging_mode() -> String {
    "trades_only".to_string()
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            enable_entry_logging: false,
            logging_mode: default_logging_mode(),
        }
    }
}

/// Stop update policy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StopUpdatePolicy {
    /// Apply stop updates on next bar
    #[default]
    ApplyNextBar,
}

/// Trade-Management-Konfiguration (MVP: `MaxHoldingTime`)
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TradeManagementConfig {
    /// Enable trade management
    #[serde(default)]
    pub enabled: bool,
    /// Stop update policy
    #[serde(default = "default_stop_update_policy")]
    pub stop_update_policy: StopUpdatePolicy,
    /// Trade management rules
    #[serde(default)]
    pub rules: TradeManagementRulesConfig,
}

fn default_stop_update_policy() -> StopUpdatePolicy {
    StopUpdatePolicy::ApplyNextBar
}

/// Trade management rules configuration
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct TradeManagementRulesConfig {
    /// Max holding time configuration
    #[serde(default)]
    pub max_holding_time: MaxHoldingTimeConfig,
}

/// Max holding time configuration
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MaxHoldingTimeConfig {
    /// Enable max holding time rule
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// Maximum holding time in minutes
    #[serde(default)]
    pub max_holding_minutes: Option<u64>,
    /// Nur für bestimmte Szenarien (leer = alle)
    #[serde(default)]
    pub only_scenarios: Vec<u8>,
}

fn default_true() -> bool {
    true
}

impl Default for MaxHoldingTimeConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_holding_minutes: None,
            only_scenarios: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FLOAT_EPS: f64 = 1e-12;

    fn assert_f64_eq(left: f64, right: f64) {
        assert!((left - right).abs() < FLOAT_EPS);
    }

    #[test]
    fn test_account_config_defaults() {
        let config = AccountConfig::default();
        assert_f64_eq(config.initial_balance, 10000.0);
        assert_eq!(config.account_currency, "EUR");
        assert_f64_eq(config.risk_per_trade, 100.0);
        assert_eq!(config.max_positions, 1);
    }

    #[test]
    fn test_costs_config_defaults() {
        let config = CostsConfig::default();
        assert!(config.enabled);
        assert_f64_eq(config.fee_multiplier, 1.0);
        assert_f64_eq(config.slippage_multiplier, 1.0);
        assert_f64_eq(config.spread_multiplier, 1.0);
        assert_f64_eq(config.pip_buffer_factor, 0.5);
    }

    #[test]
    fn test_session_config_contains_normal() {
        let session = SessionConfig {
            start: "08:00".to_string(),
            end: "17:00".to_string(),
        };

        // 8:00 = 8*3600 = 28800
        assert!(session.contains(28800));
        // 12:00 = 12*3600 = 43200
        assert!(session.contains(43200));
        // 17:00 = 17*3600 = 61200
        assert!(!session.contains(61200));
        // 07:00 = 7*3600 = 25200
        assert!(!session.contains(25200));
    }

    #[test]
    fn test_session_config_contains_cross_midnight() {
        let session = SessionConfig {
            start: "22:00".to_string(),
            end: "06:00".to_string(),
        };

        // 23:00 = 23*3600 = 82800
        assert!(session.contains(82800));
        // 01:00 = 1*3600 = 3600
        assert!(session.contains(3600));
        // 12:00 = 12*3600 = 43200
        assert!(!session.contains(43200));
    }

    #[test]
    fn test_config_serde_roundtrip() {
        let config = BacktestConfig {
            schema_version: "2.0".to_string(),
            strategy_name: "TestStrategy".to_string(),
            symbol: "EURUSD".to_string(),
            start_date: "2024-01-01".to_string(),
            end_date: "2024-12-31".to_string(),
            run_mode: RunMode::Dev,
            data_mode: DataMode::Candle,
            execution_variant: ExecutionVariant::V2,
            timeframes: TimeframeConfig {
                primary: "H1".to_string(),
                additional: vec!["M15".to_string()],
                additional_source: "separate_parquet".to_string(),
            },
            warmup_bars: 500,
            rng_seed: Some(42),
            sessions: None,
            account: AccountConfig::default(),
            costs: CostsConfig::default(),
            news_filter: None,
            logging: LoggingConfig::default(),
            metrics: None,
            trade_management: None,
            strategy_parameters: serde_json::json!({"param1": 10}),
        };

        let json = serde_json::to_string(&config).unwrap();
        let deserialized: BacktestConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(config.schema_version, deserialized.schema_version);
        assert_eq!(config.strategy_name, deserialized.strategy_name);
        assert_eq!(config.warmup_bars, deserialized.warmup_bars);
    }

    #[test]
    fn test_rng_seed_default_dev_when_missing() {
        let json = serde_json::json!({
            "schema_version": "2",
            "strategy_name": "TestStrategy",
            "symbol": "EURUSD",
            "start_date": "2024-01-01",
            "end_date": "2024-12-31",
            "run_mode": "dev",
            "data_mode": "candle",
            "execution_variant": "v2",
            "timeframes": {"primary": "H1"}
        });

        let deserialized: BacktestConfig = serde_json::from_str(&json.to_string()).unwrap();

        assert_eq!(deserialized.rng_seed, Some(DEFAULT_RNG_SEED));
    }
}
