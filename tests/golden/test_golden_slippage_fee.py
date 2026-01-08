"""Golden-File Tests für Slippage- und Fee-Module.

Dieses Modul stellt sicher, dass `SlippageModel` und `FeeModel` aus
`src/backtest_engine/core/slippage_and_fee.py` deterministisch und
reproduzierbar arbeiten. Dies ist ein kritischer Gate für die
Migration zu Rust/Julia.

Invarianten:
- Identische Inputs + Seeds → identische Outputs
- Keine verborgenen Zustandsabhängigkeiten
- Rundungs-konsistent über Python-Versionen

Golden-File: tests/golden/reference/slippage_fee/slippage_fee_v1.json
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.backtest_engine.core.slippage_and_fee import FeeModel, SlippageModel
from tests.golden.conftest import (
    GOLDEN_REFERENCE_DIR,
    GoldenFileMetadata,
    compute_dict_hash,
    create_metadata,
    set_deterministic_seed,
)

# ==============================================================================
# CONSTANTS
# ==============================================================================

GOLDEN_SEED = 42
REFERENCE_FILE = GOLDEN_REFERENCE_DIR / "slippage_fee" / "slippage_fee_v1.json"


# ==============================================================================
# DATACLASS FOR GOLDEN RESULTS
# ==============================================================================


@dataclass
class GoldenSlippageFeeResult:
    """Struktur für Slippage/Fee Golden-File.

    Speichert deterministische Test-Outputs für Regressionstests.
    """

    metadata: GoldenFileMetadata
    slippage_test_cases: list[dict[str, Any]]
    slippage_results_hash: str
    fee_test_cases: list[dict[str, Any]]
    fee_results_hash: str

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["metadata"] = self.metadata.to_dict()
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GoldenSlippageFeeResult:
        metadata = GoldenFileMetadata.from_dict(data["metadata"])
        return cls(
            metadata=metadata,
            slippage_test_cases=data["slippage_test_cases"],
            slippage_results_hash=data["slippage_results_hash"],
            fee_test_cases=data["fee_test_cases"],
            fee_results_hash=data["fee_results_hash"],
        )


# ==============================================================================
# TEST DATA GENERATORS
# ==============================================================================


def generate_slippage_test_cases(seed: int) -> list[dict[str, Any]]:
    """Generiert deterministische Slippage-Testfälle.

    Args:
        seed: Seed für Reproduzierbarkeit.

    Returns:
        Liste von Testfällen mit Input-Parametern und erwarteten Outputs.
    """
    set_deterministic_seed(seed)

    test_cases = []

    # Fixe Parameter für SlippageModel
    fixed_pips = 0.5
    random_pips = 1.0
    pip_size = 0.0001  # Standard für EURUSD

    model = SlippageModel(fixed_pips=fixed_pips, random_pips=random_pips)

    # Test-Scenarios: verschiedene Preise und Richtungen
    # direction ist ein String: "long" oder "short"
    prices = [1.10000, 1.15000, 1.20000, 1.25000, 1.30000]
    directions = ["long", "short"]

    for price in prices:
        for direction in directions:
            # Pass seed directly to apply() for determinism (supports both Python and Rust backends)
            adjusted_price = model.apply(price, direction, pip_size, seed=seed)

            test_cases.append(
                {
                    "input": {
                        "price": price,
                        "direction": direction,
                        "pip_size": pip_size,
                        "fixed_pips": fixed_pips,
                        "random_pips": random_pips,
                        "seed": seed,
                    },
                    "output": {
                        "adjusted_price": round(adjusted_price, 8),
                    },
                }
            )

    return test_cases


def generate_fee_test_cases() -> list[dict[str, Any]]:
    """Generiert deterministische Fee-Testfälle.

    Fee-Berechnung ist rein deterministisch (kein Random),
    daher brauchen wir keinen Seed.

    Returns:
        Liste von Testfällen mit Input-Parametern und erwarteten Outputs.
    """
    test_cases = []

    # FeeModel Parameter
    per_million = 30.0  # $30 pro Million Notional
    lot_size = 100_000
    min_fee = 0.01

    model = FeeModel(per_million=per_million, lot_size=lot_size, min_fee=min_fee)

    # Test-Scenarios: verschiedene Volumen und Preise
    volumes_lots = [0.01, 0.1, 1.0, 5.0, 10.0]
    prices = [1.10000, 1.15000, 1.20000]
    contract_size = 100_000  # Standard Lot

    for volume in volumes_lots:
        for price in prices:
            fee = model.calculate(volume, price, contract_size)

            test_cases.append(
                {
                    "input": {
                        "volume_lots": volume,
                        "price": price,
                        "contract_size": contract_size,
                        "per_million": per_million,
                        "lot_size": lot_size,
                        "min_fee": min_fee,
                    },
                    "output": {
                        "fee": round(fee, 8),
                    },
                }
            )

    return test_cases


# ==============================================================================
# GOLDEN FILE MANAGEMENT
# ==============================================================================


def load_golden_reference() -> GoldenSlippageFeeResult | None:
    """Lädt die Golden-Referenz wenn vorhanden."""
    if not REFERENCE_FILE.exists():
        return None

    with open(REFERENCE_FILE) as f:
        data = json.load(f)

    return GoldenSlippageFeeResult.from_dict(data)


def save_golden_reference(result: GoldenSlippageFeeResult) -> Path:
    """Speichert eine neue Golden-Referenz.

    Nur zur initialen Generierung verwenden!
    """
    REFERENCE_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(REFERENCE_FILE, "w") as f:
        json.dump(result.to_dict(), f, indent=2)

    return REFERENCE_FILE


# ==============================================================================
# TESTS
# ==============================================================================


class TestSlippageModelDeterminism:
    """Deterministismus-Tests für SlippageModel."""

    def test_slippage_deterministic_with_fixed_seed(self) -> None:
        """SlippageModel muss bei gleichem Seed gleiche Ergebnisse liefern."""
        model = SlippageModel(fixed_pips=0.5, random_pips=1.0)

        results_1 = []
        for price in [1.1, 1.2, 1.3]:
            # Pass seed directly to apply() for determinism (supports both Python and Rust)
            results_1.append(model.apply(price, "long", 0.0001, seed=GOLDEN_SEED))

        results_2 = []
        for price in [1.1, 1.2, 1.3]:
            results_2.append(model.apply(price, "long", 0.0001, seed=GOLDEN_SEED))

        assert results_1 == results_2, "SlippageModel nicht deterministisch!"

    def test_slippage_direction_awareness(self) -> None:
        """Slippage muss richtungsabhängig sein (long vs short)."""
        set_deterministic_seed(GOLDEN_SEED)
        model = SlippageModel(fixed_pips=1.0, random_pips=0.0)

        price = 1.10000
        pip_size = 0.0001

        long_price = model.apply(price, "long", pip_size)
        short_price = model.apply(price, "short", pip_size)

        # long sollte teurer werden (Slippage nach oben)
        # short sollte günstiger werden (Slippage nach unten)
        assert long_price > price, "long-Slippage sollte Preis erhöhen"
        assert short_price < price, "short-Slippage sollte Preis senken"


class TestFeeModelDeterminism:
    """Deterministismus-Tests für FeeModel."""

    def test_fee_calculation_deterministic(self) -> None:
        """FeeModel muss immer gleiche Ergebnisse liefern (kein Random)."""
        model = FeeModel(per_million=30.0, lot_size=100_000, min_fee=0.01)

        fee_1 = model.calculate(1.0, 1.10000, 100_000)
        fee_2 = model.calculate(1.0, 1.10000, 100_000)

        assert fee_1 == fee_2, "FeeModel nicht deterministisch!"

    def test_fee_minimum_enforced(self) -> None:
        """Minimale Fee muss eingehalten werden."""
        min_fee = 1.00
        model = FeeModel(per_million=30.0, lot_size=100_000, min_fee=min_fee)

        # Sehr kleines Volumen sollte min_fee liefern
        fee = model.calculate(0.001, 1.0, 100_000)
        assert fee >= min_fee, f"Fee {fee} unter Minimum {min_fee}"


class TestGoldenFileValidation:
    """Validierung gegen Golden-File Referenz."""

    def test_slippage_matches_golden_reference(self) -> None:
        """Slippage-Ergebnisse müssen mit Golden-Referenz übereinstimmen."""
        reference = load_golden_reference()

        if reference is None:
            pytest.skip(
                "Golden-Referenz nicht vorhanden. "
                "Generiere mit: pytest --generate-golden"
            )

        # Generiere aktuelle Testfälle
        current_cases = generate_slippage_test_cases(GOLDEN_SEED)
        current_hash = compute_dict_hash({"cases": current_cases})

        assert current_hash == reference.slippage_results_hash, (
            f"Slippage-Hash Mismatch!\n"
            f"Erwartet: {reference.slippage_results_hash}\n"
            f"Aktuell:  {current_hash}\n"
            f"Dies deutet auf nicht-deterministisches Verhalten hin."
        )

    def test_fee_matches_golden_reference(self) -> None:
        """Fee-Ergebnisse müssen mit Golden-Referenz übereinstimmen."""
        reference = load_golden_reference()

        if reference is None:
            pytest.skip(
                "Golden-Referenz nicht vorhanden. "
                "Generiere mit: pytest --generate-golden"
            )

        current_cases = generate_fee_test_cases()
        current_hash = compute_dict_hash({"cases": current_cases})

        assert current_hash == reference.fee_results_hash, (
            f"Fee-Hash Mismatch!\n"
            f"Erwartet: {reference.fee_results_hash}\n"
            f"Aktuell:  {current_hash}\n"
            f"Dies deutet auf Berechnungsänderungen hin."
        )

    def test_all_test_cases_match_individually(self) -> None:
        """Jeder einzelne Testfall muss exakt übereinstimmen."""
        reference = load_golden_reference()

        if reference is None:
            pytest.skip("Golden-Referenz nicht vorhanden.")

        # Prüfe jeden Slippage-Testfall
        current_slippage = generate_slippage_test_cases(GOLDEN_SEED)
        for i, (current, expected) in enumerate(
            zip(current_slippage, reference.slippage_test_cases, strict=True)
        ):
            assert current == expected, (
                f"Slippage Testfall {i} Mismatch:\n"
                f"Erwartet: {expected}\n"
                f"Aktuell:  {current}"
            )

        # Prüfe jeden Fee-Testfall
        current_fee = generate_fee_test_cases()
        for i, (current, expected) in enumerate(
            zip(current_fee, reference.fee_test_cases, strict=True)
        ):
            assert current == expected, (
                f"Fee Testfall {i} Mismatch:\n"
                f"Erwartet: {expected}\n"
                f"Aktuell:  {current}"
            )


# ==============================================================================
# GOLDEN FILE GENERATION (manual, not run in CI)
# ==============================================================================


def generate_golden_reference() -> GoldenSlippageFeeResult:
    """Generiert eine neue Golden-Referenz.

    WARNUNG: Nur manuell aufrufen, wenn Änderungen beabsichtigt sind!
    """
    slippage_cases = generate_slippage_test_cases(GOLDEN_SEED)
    fee_cases = generate_fee_test_cases()

    return GoldenSlippageFeeResult(
        metadata=create_metadata(
            seed=GOLDEN_SEED,
            description=(
                "Golden-Reference für SlippageModel und FeeModel. "
                "Generiert für Migration Readiness Validation (Pilot Module)."
            ),
        ),
        slippage_test_cases=slippage_cases,
        slippage_results_hash=compute_dict_hash({"cases": slippage_cases}),
        fee_test_cases=fee_cases,
        fee_results_hash=compute_dict_hash({"cases": fee_cases}),
    )


if __name__ == "__main__":
    # Manuelle Generierung: python -m tests.golden.test_golden_slippage_fee
    print("Generating Golden Reference for Slippage/Fee...")
    result = generate_golden_reference()
    path = save_golden_reference(result)
    print(f"Saved to: {path}")
    print(f"Slippage Hash: {result.slippage_results_hash}")
    print(f"Fee Hash: {result.fee_results_hash}")
