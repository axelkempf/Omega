"""
Golden-File Test Configuration und Utilities.

Dieses Modul enthält:
1. pytest-Fixtures für Golden-File Tests
2. Utilities zum Generieren und Vergleichen von Referenz-Outputs
3. Hilfsfunktionen für deterministisches Seeding
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import warnings
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union

import numpy as np
import pandas as pd
import pytest

# Pfad zu Referenz-Dateien
GOLDEN_REFERENCE_DIR = Path(__file__).parent / "reference"
GOLDEN_REFERENCE_DIR.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# DATENKLASSEN FÜR GOLDEN-FILE STRUKTUR
# ==============================================================================


@dataclass
class GoldenFileMetadata:
    """Metadata für Golden-File Referenz."""

    created_at: str
    python_version: str
    numpy_version: str
    pandas_version: str
    seed: int
    description: str
    file_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> GoldenFileMetadata:
        return cls(**data)


@dataclass
class GoldenBacktestResult:
    """Struktur für Backtest Golden-File."""

    metadata: GoldenFileMetadata
    summary_metrics: Dict[str, Any]
    trade_count: int
    trade_hashes: List[str]  # Hash jedes Trades für schnellen Vergleich
    equity_curve_hash: str
    final_equity: float
    total_pnl: float

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["metadata"] = self.metadata.to_dict()
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> GoldenBacktestResult:
        metadata = GoldenFileMetadata.from_dict(data["metadata"])
        return cls(
            metadata=metadata,
            summary_metrics=data["summary_metrics"],
            trade_count=data["trade_count"],
            trade_hashes=data["trade_hashes"],
            equity_curve_hash=data["equity_curve_hash"],
            final_equity=data["final_equity"],
            total_pnl=data["total_pnl"],
        )


@dataclass
class GoldenOptimizerResult:
    """Struktur für Optimizer Golden-File."""

    metadata: GoldenFileMetadata
    best_params: Dict[str, Any]
    best_score: float
    n_trials: int
    param_ranges: Dict[str, Any]
    top_n_hashes: List[str]  # Hash der Top-N Ergebnisse

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["metadata"] = self.metadata.to_dict()
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> GoldenOptimizerResult:
        metadata = GoldenFileMetadata.from_dict(data["metadata"])
        return cls(
            metadata=metadata,
            best_params=data["best_params"],
            best_score=data["best_score"],
            n_trials=data["n_trials"],
            param_ranges=data["param_ranges"],
            top_n_hashes=data["top_n_hashes"],
        )


@dataclass
class GoldenRatingResult:
    """Struktur für Rating-Module Golden-File.

    Speichert bewusst keine großen Rohdaten (z.B. komplette DataFrames), sondern
    stabile Scalar-Outputs und Hashes abgeleiteter Artefakte.
    """

    metadata: GoldenFileMetadata
    outputs: Dict[str, Any]
    outputs_hash: str

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["metadata"] = self.metadata.to_dict()
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> GoldenRatingResult:
        metadata = GoldenFileMetadata.from_dict(data["metadata"])
        return cls(
            metadata=metadata,
            outputs=data["outputs"],
            outputs_hash=data["outputs_hash"],
        )


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================


def set_deterministic_seed(seed: int) -> None:
    """
    Setzt alle relevanten Seeds für deterministische Reproduktion.

    Args:
        seed: Integer-Seed für RNG.
    """
    random.seed(seed)
    np.random.seed(seed % (2**32))  # numpy erwartet 32-bit Seed

    # Für torch wenn verfügbar
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True  # type: ignore
        torch.backends.cudnn.benchmark = False  # type: ignore
    except ImportError:
        pass

    # Environment-Variablen für weitere Determinismus
    os.environ["PYTHONHASHSEED"] = str(seed)


def compute_dict_hash(data: Dict[str, Any], *, precision: int = 8) -> str:
    """
    Berechnet einen stabilen Hash für ein Dictionary.

    Args:
        data: Dictionary zu hashen.
        precision: Rundungspräzision für Floats.

    Returns:
        SHA256-Hash als Hex-String.
    """

    def _normalize(obj: Any) -> Any:
        if isinstance(obj, float):
            if math.isnan(obj) or math.isinf(obj):
                return str(obj)
            return round(obj, precision)
        if isinstance(obj, dict):
            return {k: _normalize(v) for k, v in sorted(obj.items())}
        if isinstance(obj, (list, tuple)):
            return [_normalize(x) for x in obj]
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            val = float(obj)
            if math.isnan(val) or math.isinf(val):
                return str(val)
            return round(val, precision)
        if isinstance(obj, np.ndarray):
            return _normalize(obj.tolist())
        if isinstance(obj, (datetime, pd.Timestamp)):
            return obj.isoformat()
        return obj

    normalized = _normalize(data)
    json_str = json.dumps(normalized, sort_keys=True)
    return hashlib.sha256(json_str.encode()).hexdigest()


def compute_dataframe_hash(
    df: pd.DataFrame, *, precision: int = 8, columns: Optional[List[str]] = None
) -> str:
    """
    Berechnet einen stabilen Hash für einen DataFrame.

    Args:
        df: DataFrame zu hashen.
        precision: Rundungspräzision für Floats.
        columns: Optionale Spaltenauswahl.

    Returns:
        SHA256-Hash als Hex-String.
    """
    if columns:
        df = df[columns]

    # Runde numerische Spalten
    df_copy = df.copy()
    for col in df_copy.select_dtypes(include=[np.number]).columns:
        df_copy[col] = df_copy[col].round(precision)

    # Konvertiere zu String für Hashing
    csv_str = df_copy.to_csv(index=False)
    return hashlib.sha256(csv_str.encode()).hexdigest()


def create_metadata(seed: int, description: str) -> GoldenFileMetadata:
    """Erstellt Metadata für eine Golden-File."""
    import sys

    return GoldenFileMetadata(
        created_at=datetime.now(UTC).isoformat(),
        python_version=sys.version.split()[0],
        numpy_version=np.__version__,
        pandas_version=pd.__version__,
        seed=seed,
        description=description,
    )


class GoldenFileComparisonError(AssertionError):
    """Exception für Golden-File Vergleichsfehler."""

    def __init__(self, message: str, details: Dict[str, Any]):
        super().__init__(message)
        self.details = details


# ==============================================================================
# GOLDEN FILE MANAGER
# ==============================================================================


class GoldenFileManager:
    """
    Manager für Golden-File Operationen.

    Unterstützt:
    - Speichern und Laden von Referenz-Files
    - Vergleichen von aktuellen Outputs mit Referenzen
    - Automatische Regeneration bei Bedarf
    """

    def __init__(self, reference_dir: Path = GOLDEN_REFERENCE_DIR):
        self.reference_dir = reference_dir
        self.reference_dir.mkdir(parents=True, exist_ok=True)

    def _get_reference_path(self, name: str, category: str = "backtest") -> Path:
        """Berechnet den Pfad für eine Referenz-Datei."""
        category_dir = self.reference_dir / category
        category_dir.mkdir(parents=True, exist_ok=True)
        return category_dir / f"{name}.json"

    def save_backtest_reference(self, name: str, result: GoldenBacktestResult) -> Path:
        """Speichert eine Backtest-Referenz."""
        path = self._get_reference_path(name, "backtest")

        # Berechne Gesamt-Hash
        result.metadata.file_hash = compute_dict_hash(result.to_dict())

        with open(path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

        return path

    def load_backtest_reference(self, name: str) -> Optional[GoldenBacktestResult]:
        """Lädt eine Backtest-Referenz."""
        path = self._get_reference_path(name, "backtest")
        if not path.exists():
            return None

        with open(path) as f:
            data = json.load(f)

        return GoldenBacktestResult.from_dict(data)

    def save_optimizer_reference(
        self, name: str, result: GoldenOptimizerResult
    ) -> Path:
        """Speichert eine Optimizer-Referenz."""
        path = self._get_reference_path(name, "optimizer")

        # Berechne Gesamt-Hash
        result.metadata.file_hash = compute_dict_hash(result.to_dict())

        with open(path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

        return path

    def load_optimizer_reference(self, name: str) -> Optional[GoldenOptimizerResult]:
        """Lädt eine Optimizer-Referenz."""
        path = self._get_reference_path(name, "optimizer")
        if not path.exists():
            return None

        with open(path) as f:
            data = json.load(f)

        return GoldenOptimizerResult.from_dict(data)

    def save_rating_reference(self, name: str, result: GoldenRatingResult) -> Path:
        """Speichert eine Rating-Referenz."""
        path = self._get_reference_path(name, "rating")

        # Hash primär über die Outputs (ohne volatile Metadaten wie created_at).
        result.outputs_hash = compute_dict_hash(result.outputs)

        # Optional: Gesamt-Hash (inkl. Metadaten) für Debug / Datei-Integrität.
        result.metadata.file_hash = compute_dict_hash(result.to_dict())

        with open(path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

        return path

    def load_rating_reference(self, name: str) -> Optional[GoldenRatingResult]:
        """Lädt eine Rating-Referenz."""
        path = self._get_reference_path(name, "rating")
        if not path.exists():
            return None

        with open(path) as f:
            data = json.load(f)

        return GoldenRatingResult.from_dict(data)

    def compare_rating_results(
        self,
        name: str,
        current: GoldenRatingResult,
        *,
        metric_tolerance: float = 1e-8,
    ) -> Dict[str, Any]:
        """Vergleicht aktuelles Rating-Ergebnis mit Referenz."""
        reference = self.load_rating_reference(name)

        if reference is None:
            return {
                "status": "no_reference",
                "message": f"No reference found for '{name}'",
            }

        # Recompute to avoid caller mistakes.
        current.outputs_hash = compute_dict_hash(current.outputs)

        differences: Dict[str, Any] = {}

        if current.outputs_hash != reference.outputs_hash:
            differences["outputs_hash"] = {
                "expected": reference.outputs_hash,
                "actual": current.outputs_hash,
            }

            # Minimaler, stabiler Top-Level-Diff für Debugbarkeit.
            exp = reference.outputs or {}
            cur = current.outputs or {}
            keys = sorted(set(exp.keys()) | set(cur.keys()))
            per_key: Dict[str, Any] = {}
            for k in keys:
                if k not in exp:
                    per_key[k] = {"expected": "<missing>", "actual": cur.get(k)}
                    continue
                if k not in cur:
                    per_key[k] = {"expected": exp.get(k), "actual": "<missing>"}
                    continue
                ev = exp.get(k)
                cv = cur.get(k)
                if isinstance(ev, (int, float)) and isinstance(cv, (int, float)):
                    diff = abs(float(ev) - float(cv))
                    if diff > float(metric_tolerance):
                        per_key[k] = {
                            "expected": float(ev),
                            "actual": float(cv),
                            "diff": float(diff),
                        }
                elif isinstance(ev, dict) and isinstance(cv, dict):
                    eh = compute_dict_hash(ev)
                    ch = compute_dict_hash(cv)
                    if eh != ch:
                        per_key[k] = {
                            "expected_hash": eh,
                            "actual_hash": ch,
                        }
                elif ev != cv:
                    per_key[k] = {"expected": ev, "actual": cv}

            if per_key:
                differences["outputs_top_level"] = per_key

        if differences:
            raise GoldenFileComparisonError(
                f"Golden file comparison failed for '{name}'",
                differences,
            )

        return {"status": "match", "reference_name": name}

    def compare_backtest_results(
        self,
        name: str,
        current: GoldenBacktestResult,
        *,
        metric_tolerance: float = 1e-6,
        strict_trades: bool = True,
    ) -> Dict[str, Any]:
        """
        Vergleicht aktuelles Backtest-Ergebnis mit Referenz.

        Args:
            name: Name der Referenz.
            current: Aktuelles Ergebnis.
            metric_tolerance: Toleranz für numerische Metriken.
            strict_trades: Ob Trade-Hashes exakt übereinstimmen müssen.

        Returns:
            Dict mit Vergleichsergebnis.

        Raises:
            GoldenFileComparisonError: Bei Abweichungen.
        """
        reference = self.load_backtest_reference(name)

        if reference is None:
            return {
                "status": "no_reference",
                "message": f"No reference found for '{name}'",
            }

        differences: Dict[str, Any] = {}

        # Trade Count
        if current.trade_count != reference.trade_count:
            differences["trade_count"] = {
                "expected": reference.trade_count,
                "actual": current.trade_count,
            }

        # Final Equity (mit Toleranz)
        equity_diff = abs(current.final_equity - reference.final_equity)
        if equity_diff > metric_tolerance:
            differences["final_equity"] = {
                "expected": reference.final_equity,
                "actual": current.final_equity,
                "diff": equity_diff,
            }

        # Total PnL (mit Toleranz)
        pnl_diff = abs(current.total_pnl - reference.total_pnl)
        if pnl_diff > metric_tolerance:
            differences["total_pnl"] = {
                "expected": reference.total_pnl,
                "actual": current.total_pnl,
                "diff": pnl_diff,
            }

        # Trade Hashes (wenn strict)
        if strict_trades and current.trade_hashes != reference.trade_hashes:
            differences["trade_hashes"] = {
                "expected_count": len(reference.trade_hashes),
                "actual_count": len(current.trade_hashes),
                "mismatches": sum(
                    1
                    for a, b in zip(current.trade_hashes, reference.trade_hashes)
                    if a != b
                ),
            }

        # Equity Curve Hash
        if current.equity_curve_hash != reference.equity_curve_hash:
            differences["equity_curve_hash"] = {
                "expected": reference.equity_curve_hash[:16] + "...",
                "actual": current.equity_curve_hash[:16] + "...",
            }

        # Summary Metrics (wichtige Metriken prüfen)
        for key in [
            "net_profit_eur",
            "winrate_percent",
            "profit_factor",
            "sharpe_trade",
        ]:
            if key in reference.summary_metrics and key in current.summary_metrics:
                ref_val = reference.summary_metrics[key]
                cur_val = current.summary_metrics[key]
                if isinstance(ref_val, (int, float)) and isinstance(
                    cur_val, (int, float)
                ):
                    if abs(ref_val - cur_val) > metric_tolerance:
                        if "summary_metrics" not in differences:
                            differences["summary_metrics"] = {}
                        differences["summary_metrics"][key] = {
                            "expected": ref_val,
                            "actual": cur_val,
                        }

        if differences:
            raise GoldenFileComparisonError(
                f"Golden file comparison failed for '{name}'",
                differences,
            )

        return {"status": "match", "reference_name": name}

    def compare_optimizer_results(
        self,
        name: str,
        current: GoldenOptimizerResult,
        *,
        score_tolerance: float = 1e-6,
    ) -> Dict[str, Any]:
        """
        Vergleicht aktuelles Optimizer-Ergebnis mit Referenz.

        Args:
            name: Name der Referenz.
            current: Aktuelles Ergebnis.
            score_tolerance: Toleranz für Scores.

        Returns:
            Dict mit Vergleichsergebnis.

        Raises:
            GoldenFileComparisonError: Bei Abweichungen.
        """
        reference = self.load_optimizer_reference(name)

        if reference is None:
            return {
                "status": "no_reference",
                "message": f"No reference found for '{name}'",
            }

        differences: Dict[str, Any] = {}

        # Best Score
        score_diff = abs(current.best_score - reference.best_score)
        if score_diff > score_tolerance:
            differences["best_score"] = {
                "expected": reference.best_score,
                "actual": current.best_score,
                "diff": score_diff,
            }

        # Best Params
        if current.best_params != reference.best_params:
            differences["best_params"] = {
                "expected": reference.best_params,
                "actual": current.best_params,
            }

        # Trial Count
        if current.n_trials != reference.n_trials:
            differences["n_trials"] = {
                "expected": reference.n_trials,
                "actual": current.n_trials,
            }

        # Top-N Hashes
        if current.top_n_hashes != reference.top_n_hashes:
            differences["top_n_hashes"] = {
                "expected_count": len(reference.top_n_hashes),
                "actual_count": len(current.top_n_hashes),
            }

        if differences:
            raise GoldenFileComparisonError(
                f"Golden file comparison failed for '{name}'",
                differences,
            )

        return {"status": "match", "reference_name": name}


# ==============================================================================
# PYTEST FIXTURES
# ==============================================================================


@pytest.fixture
def golden_manager() -> GoldenFileManager:
    """Fixture für GoldenFileManager."""
    return GoldenFileManager()


@pytest.fixture
def deterministic_seed() -> int:
    """Standard-Seed für deterministische Tests."""
    return 42


@pytest.fixture
def set_seed(deterministic_seed: int):
    """Fixture das den Seed vor jedem Test setzt."""
    set_deterministic_seed(deterministic_seed)
    yield deterministic_seed
    # Cleanup nicht nötig


@pytest.fixture
def golden_backtest_config() -> Dict[str, Any]:
    """
    Minimale Backtest-Konfiguration für Golden-File Tests.

    Diese Konfig ist vereinfacht für schnelle, deterministische Tests.
    Sie verwendet einen festen Seed und kurzen Zeitraum.
    """
    return {
        "symbol": "EURUSD",
        "timeframes": {"primary": "M15", "additional": []},
        "mode": "candle",
        "initial_balance": 100000.0,
        "risk_per_trade": 100.0,
        "account_currency": "USD",
        "start_date": "2024-01-01",
        "end_date": "2024-01-31",
        "warmup_bars": 100,
        "execution": {"random_seed": 42},
        "reporting": {"dev_mode": True, "dev_seed": 42},
    }


@pytest.fixture
def regenerate_golden_files(request) -> bool:
    """
    Fixture zum Regenerieren von Golden-Files.

    Verwende: pytest --regenerate-golden-files
    """
    return request.config.getoption("--regenerate-golden-files", default=False)


def pytest_addoption(parser):
    """
    Pytest-Hook zum Registrieren zusätzlicher CLI-Optionen.

    Fügt die Option ``--regenerate-golden-files`` hinzu, mit der Tests anweisen
    können, Golden-Reference-Dateien neu zu erzeugen, statt ausschließlich
    gegen bestehende Dateien zu vergleichen. Diese Option wird typischerweise
    über die ``regenerate_golden_files``-Fixture im Testcode ausgewertet.
    """
    parser.addoption(
        "--regenerate-golden-files",
        action="store_true",
        default=False,
        help="Regenerate golden reference files instead of comparing",
    )


# ==============================================================================
# CONVENIENCE FUNCTIONS FÜR TESTS
# ==============================================================================


def assert_golden_match(
    manager: GoldenFileManager,
    name: str,
    current_result: Union[
        GoldenBacktestResult, GoldenOptimizerResult, GoldenRatingResult
    ],
    *,
    regenerate: bool = False,
) -> None:
    """
    Assertiert dass ein Ergebnis mit seiner Golden-File übereinstimmt.

    Args:
        manager: GoldenFileManager Instanz.
        name: Name der Referenz.
        current_result: Aktuelles Ergebnis.
        regenerate: Ob die Referenz neu generiert werden soll.
    """
    if isinstance(current_result, GoldenBacktestResult):
        if regenerate:
            manager.save_backtest_reference(name, current_result)
            pytest.skip(f"Regenerated golden file: {name}")
        else:
            result = manager.compare_backtest_results(name, current_result)
            if result["status"] == "no_reference":
                manager.save_backtest_reference(name, current_result)
                warnings.warn(f"Created new golden file: {name}")

    elif isinstance(current_result, GoldenOptimizerResult):
        if regenerate:
            manager.save_optimizer_reference(name, current_result)
            pytest.skip(f"Regenerated golden file: {name}")
        else:
            result = manager.compare_optimizer_results(name, current_result)
            if result["status"] == "no_reference":
                manager.save_optimizer_reference(name, current_result)
                warnings.warn(f"Created new golden file: {name}")

    elif isinstance(current_result, GoldenRatingResult):
        if regenerate:
            manager.save_rating_reference(name, current_result)
            pytest.skip(f"Regenerated golden file: {name}")
        else:
            result = manager.compare_rating_results(name, current_result)
            if result["status"] == "no_reference":
                manager.save_rating_reference(name, current_result)
                warnings.warn(f"Created new golden file: {name}")
