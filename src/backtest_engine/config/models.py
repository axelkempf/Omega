from __future__ import annotations

from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TimeframesConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    primary: str = "M15"
    additional: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _normalize(self) -> "TimeframesConfig":
        primary = (self.primary or "").upper()
        additional = [str(tf).upper() for tf in (self.additional or []) if tf]

        # De-duplicate while keeping order.
        seen: set[str] = set()
        deduped: list[str] = []
        for tf in additional:
            if tf == primary:
                continue
            if tf in seen:
                continue
            seen.add(tf)
            deduped.append(tf)

        # Normalize in-place (mutable model) to ensure stable downstream behavior.
        self.primary = primary or "M15"
        self.additional = deduped
        return self


class TimestampAlignmentConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    entry_timestamp_mode: Literal["open", "close"] = "open"
    additional_tfs_mode: Literal["carry_forward", "strict"] = "carry_forward"

    stale_bar_limit_bars: int = 0
    max_missing_ratio: float = 0.0

    use_previous_completed_higher_tf: bool = True
    diagnostics: bool = True

    normalize_to_timeframe: bool = False

    higher_tf_timestamps_mode: dict[str, Literal["open", "close"]] = Field(
        default_factory=dict
    )


class RatesConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    mode: Literal["static", "timeseries", "composite"] = "timeseries"
    timeframe: str | None = None
    pairs: list[str] = Field(default_factory=list)

    use_price: Literal["open", "high", "low", "close"] = "close"
    stale_limit_bars: int = 2
    strict: bool = True


class ExecutionConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    slippage_multiplier: float = 1.0
    fee_multiplier: float = 1.0
    spread_multiplier: float = 1.0

    random_seed: int | None = None


class ProfilingConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    enabled: bool = False


class FeesConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    per_million: float = 0.0
    lot_size: float = 100_000.0
    min_fee: float = 0.0


class SlippageConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    fixed_pips: float = 0.0
    random_pips: float = 0.0


class StrategyConfig(BaseModel):
    """Strategy configuration used by both single- and multi-strategy backtests."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    # Multi-strategy only
    name: str | None = None

    module: str
    class_name: str = Field(alias="class")

    parameters: dict[str, Any] = Field(default_factory=dict)

    # Required in tick + multi-strategy situations, optional otherwise.
    symbol: str | None = None

    session_filter: dict[str, Any] | None = None
    anchored_session_filter: dict[str, Any] | None = None


class BacktestConfig(BaseModel):
    """Typed config model for backtest JSON configs.

    The model is intentionally forward-compatible: unknown keys are accepted and
    preserved (extra="allow"), so older/newer config files keep working.
    """

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    start_date: date
    end_date: date

    mode: Literal["candle", "tick"] = "candle"

    symbol: str | None = None
    multi_symbols: dict[str, list[str]] | None = None

    timeframes: TimeframesConfig = Field(default_factory=TimeframesConfig)

    warmup_bars: int = 500

    enable_entry_logging: bool = False
    logging_mode: str = "trades_only"

    initial_balance: float = 10_000.0
    risk_per_trade: float = 100.0

    account_currency: str = "EUR"

    rates: RatesConfig | None = None
    rates_static: dict[str, float] | None = None

    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    profiling: ProfilingConfig = Field(default_factory=ProfilingConfig)

    fees: FeesConfig | None = None
    slippage: SlippageConfig | None = None

    execution_costs_file: str | None = None
    symbol_specs_file: str | None = None

    # Global wrappers (also supported per strategy)
    session_filter: dict[str, Any] | None = None
    anchored_session_filter: dict[str, Any] | None = None

    strategy: StrategyConfig | None = None
    strategies: list[StrategyConfig] | None = None

    @model_validator(mode="after")
    def _validate_semantics(self) -> "BacktestConfig":
        if self.start_date > self.end_date:
            raise ValueError("start_date muss <= end_date sein")

        has_single = self.strategy is not None
        has_multi = bool(self.strategies)
        if has_single == has_multi:
            raise ValueError(
                "Genau eines von 'strategy' oder 'strategies' muss gesetzt sein"
            )

        if self.mode == "tick":
            if self.multi_symbols:
                # tick + multi symbols requires symbol keys; timeframes are per symbol in config
                if not isinstance(self.multi_symbols, dict) or not self.multi_symbols:
                    raise ValueError("multi_symbols darf im Tick-Modus nicht leer sein")
            else:
                if not self.symbol:
                    raise ValueError("symbol ist im Tick-Modus Pflicht")

            if self.strategies:
                missing = [
                    s.name or s.class_name for s in self.strategies if not s.symbol
                ]
                if missing:
                    raise ValueError(
                        "Im Tick-Modus muss jede Strategie ein 'symbol' enthalten: "
                        + ", ".join(map(str, missing))
                    )

        if self.mode == "candle":
            if not (self.symbol or self.multi_symbols):
                raise ValueError(
                    "Entweder 'symbol' oder 'multi_symbols' muss gesetzt sein"
                )

        return self

    def to_legacy_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict compatible with existing dict-based code."""

        return self.model_dump(mode="json", by_alias=True, exclude_none=True)
