"""Tests für Arrow Schema Registry und Drift Detection.

Dieser Test stellt sicher, dass Arrow-Schemas zwischen Python, Rust und Julia
konsistent bleiben. Schema-Änderungen müssen explizit dokumentiert werden.

CI-Guardrail: Schlägt fehl bei undokumentierten Schema-Änderungen.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

# Skip if pyarrow not available (should not happen in normal dev env)
pa = pytest.importorskip("pyarrow")

from src.shared.arrow_schemas import (
    PYARROW_AVAILABLE,
    SCHEMA_REGISTRY,
    SCHEMA_REGISTRY_VERSION,
    get_all_schema_fingerprints,
    get_schema_fingerprint,
    validate_schema_registry,
)

# Path to golden fingerprints file
GOLDEN_FINGERPRINTS_PATH = (
    Path(__file__).parent.parent / "reports" / "schema_fingerprints.json"
)


class TestSchemaFingerprinting:
    """Tests für Schema-Fingerprinting."""

    def test_fingerprint_determinism(self) -> None:
        """Fingerprint muss bei mehrfachem Aufruf identisch sein."""
        for name, schema in SCHEMA_REGISTRY.items():
            if schema is None:
                continue
            fp1 = get_schema_fingerprint(schema)
            fp2 = get_schema_fingerprint(schema)
            assert fp1 == fp2, f"Non-deterministic fingerprint for {name}"

    def test_fingerprint_format(self) -> None:
        """Fingerprint muss SHA-256 hex format haben."""
        for name, schema in SCHEMA_REGISTRY.items():
            if schema is None:
                continue
            fp = get_schema_fingerprint(schema)
            assert len(fp) == 64, f"Invalid fingerprint length for {name}"
            assert all(
                c in "0123456789abcdef" for c in fp
            ), f"Invalid hex chars in {name}"

    def test_all_schemas_have_fingerprints(self) -> None:
        """Alle registrierten Schemas müssen Fingerprints haben."""
        fps = get_all_schema_fingerprints()
        expected_schemas = {
            "ohlcv",
            "trade_signal",
            "position",
            "indicator",
            "rating_score",
            "equity_curve",
        }

        for schema_name in expected_schemas:
            assert schema_name in fps, f"Missing fingerprint for {schema_name}"


class TestSchemaRegistry:
    """Tests für Schema Registry."""

    def test_registry_version_format(self) -> None:
        """Registry-Version muss SemVer Format haben."""
        parts = SCHEMA_REGISTRY_VERSION.split(".")
        assert len(parts) == 3, "Version must be MAJOR.MINOR.PATCH"
        assert all(part.isdigit() for part in parts), "Version parts must be numeric"

    def test_all_schemas_registered(self) -> None:
        """Alle bekannten Schemas müssen registriert sein."""
        expected = {
            "ohlcv",
            "trade_signal",
            "position",
            "indicator",
            "rating_score",
            "equity_curve",
        }
        actual = set(SCHEMA_REGISTRY.keys())

        missing = expected - actual
        assert not missing, f"Missing schemas in registry: {missing}"

    def test_schemas_have_fields(self) -> None:
        """Alle Schemas müssen mindestens ein Feld haben."""
        for name, schema in SCHEMA_REGISTRY.items():
            if schema is None:
                continue
            assert len(schema) > 0, f"Schema {name} has no fields"


class TestSchemaValidation:
    """Tests für Schema-Validierung."""

    def test_validate_matching_fingerprints(self) -> None:
        """Validation sollte bei korrekten Fingerprints keine Fehler haben."""
        current = get_all_schema_fingerprints()
        errors = validate_schema_registry(current)
        assert errors == [], f"Unexpected errors: {errors}"

    def test_validate_detects_drift(self) -> None:
        """Validation muss Schema-Drift erkennen."""
        fake_expected = {"ohlcv": "0" * 64}  # Wrong fingerprint
        errors = validate_schema_registry(fake_expected)

        assert any("drift detected" in e.lower() for e in errors), "Should detect drift"

    def test_validate_detects_missing_schema(self) -> None:
        """Validation muss fehlende Schemas erkennen."""
        expected_with_extra = get_all_schema_fingerprints()
        expected_with_extra["nonexistent_schema"] = "a" * 64

        errors = validate_schema_registry(expected_with_extra)
        assert any(
            "missing schema" in e.lower() for e in errors
        ), "Should detect missing"

    def test_validate_detects_new_schema(self) -> None:
        """Validation muss neue Schemas erkennen."""
        partial_expected = {"ohlcv": get_all_schema_fingerprints()["ohlcv"]}

        errors = validate_schema_registry(partial_expected)
        # Should report new schemas not in expected
        assert any(
            "new schema" in e.lower() for e in errors
        ), "Should detect new schemas"


class TestGoldenFingerprints:
    """CI-Guardrail: Vergleich gegen gespeicherte Fingerprints."""

    @pytest.fixture
    def golden_fingerprints(self) -> dict[str, str] | None:
        """Lade golden fingerprints falls vorhanden."""
        if not GOLDEN_FINGERPRINTS_PATH.exists():
            return None
        with open(GOLDEN_FINGERPRINTS_PATH) as f:
            return json.load(f)

    def test_no_schema_drift(self, golden_fingerprints: dict[str, str] | None) -> None:
        """CI-Test: Schema darf nicht von golden state abweichen.

        Bei Fehlschlag:
        1. Prüfen ob Schema-Änderung gewollt war
        2. Wenn ja: Golden-File manuell aktualisieren
        3. Wenn nein: Schema-Änderung rückgängig machen
        """
        if golden_fingerprints is None:
            pytest.skip(
                f"Golden fingerprints file not found at {GOLDEN_FINGERPRINTS_PATH}. "
                'Create with: python -c "from src.shared.arrow_schemas import get_all_schema_fingerprints; '
                'import json; print(json.dumps(get_all_schema_fingerprints(), indent=2))"'
            )

        errors = validate_schema_registry(golden_fingerprints)

        if errors:
            error_msg = (
                "Schema drift detected!\n\n"
                + "\n".join(f"  - {e}" for e in errors)
                + "\n\n"
                "If schema change was intentional:\n"
                "  1. Update docs/adr/ADR-0002-serialization-format.md\n"
                "  2. Update FFI specs in docs/ffi/\n"
                "  3. Update reports/schema_fingerprints.json manually\n"
                "  4. Commit the updated schema_fingerprints.json\n"
            )
            pytest.fail(error_msg)
