"""FFI ErrorCode Synchronization Test.

Verifiziert, dass die Python-ErrorCode-Enum und die Rust-ErrorCode-Enum
exakt synchronisiert sind. Dies ist ein kritisches Gate für FFI-Interoperabilität.

Dieser Test importiert beide Seiten und vergleicht die Werte direkt.
Bei Abweichungen schlägt der Test fehl und zeigt genau an, welche Codes
nicht übereinstimmen.

FFI Contract: src/shared/error_codes.py <-> src/rust_modules/omega_rust/src/error.rs
"""

from __future__ import annotations

import pytest

from src.shared.error_codes import ErrorCode, error_category, is_recoverable


class TestErrorCodePythonIntegrity:
    """Tests für Python ErrorCode Integrität."""

    def test_all_codes_have_unique_values(self) -> None:
        """Alle ErrorCodes müssen eindeutige Integer-Werte haben."""
        seen_values: dict[int, str] = {}
        duplicates: list[tuple[str, str, int]] = []

        for code in ErrorCode:
            if code.value in seen_values:
                duplicates.append((code.name, seen_values[code.value], code.value))
            else:
                seen_values[code.value] = code.name

        assert not duplicates, f"Doppelte ErrorCode-Werte gefunden: {duplicates}"

    def test_code_ranges_are_correct(self) -> None:
        """ErrorCodes müssen in den dokumentierten Bereichen liegen."""
        code_ranges = {
            "OK": (0, 0),
            "VALIDATION": (1000, 1999),
            "COMPUTATION": (2000, 2999),
            "IO": (3000, 3999),
            "INTERNAL": (4000, 4999),
            "FFI": (5000, 5999),
            "RESOURCE": (6000, 6999),
        }

        for code in ErrorCode:
            if code.value == 0:
                continue

            category = error_category(code)
            if category not in code_ranges:
                pytest.fail(f"Unbekannte Kategorie {category} für {code.name}")

            low, high = code_ranges[category]
            assert low <= code.value <= high, (
                f"{code.name}={code.value} liegt außerhalb des "
                f"{category}-Bereichs [{low}, {high}]"
            )

    def test_is_recoverable_consistency(self) -> None:
        """is_recoverable muss konsistente Ergebnisse liefern."""
        for code in ErrorCode:
            result_enum = is_recoverable(code)
            result_int = is_recoverable(code.value)
            assert (
                result_enum == result_int
            ), f"is_recoverable({code.name}) != is_recoverable({code.value})"


class TestRustPythonErrorCodeSync:
    """Tests für Python/Rust ErrorCode Synchronisation.

    Diese Tests erfordern das kompilierte Rust-Modul. Falls nicht verfügbar,
    werden sie übersprungen.
    """

    @pytest.fixture(scope="class")
    def rust_error_codes(self) -> dict[str, int]:
        """Versucht die Rust ErrorCode-Konstanten zu laden."""
        try:
            from omega_rust import get_error_code_constants  # type: ignore

            return get_error_code_constants()
        except ImportError:
            pytest.skip(
                "Rust-Modul 'omega_rust' nicht verfügbar. "
                "Baue mit: maturin develop -r"
            )

    def test_rust_module_exports_error_codes(
        self, rust_error_codes: dict[str, int]
    ) -> None:
        """Rust-Modul muss get_error_code_constants exportieren."""
        assert isinstance(rust_error_codes, dict), "Muss ein Dictionary sein"
        assert len(rust_error_codes) > 0, "Dictionary darf nicht leer sein"

    def test_all_python_codes_exist_in_rust(
        self, rust_error_codes: dict[str, int]
    ) -> None:
        """Alle Python ErrorCodes müssen in Rust existieren."""
        missing_in_rust: list[str] = []

        for code in ErrorCode:
            if code.name not in rust_error_codes:
                missing_in_rust.append(code.name)

        assert not missing_in_rust, (
            f"Folgende Python-ErrorCodes fehlen in Rust:\n"
            f"{missing_in_rust}\n\n"
            f"Bitte src/rust_modules/omega_rust/src/error.rs aktualisieren."
        )

    def test_all_rust_codes_exist_in_python(
        self, rust_error_codes: dict[str, int]
    ) -> None:
        """Alle Rust ErrorCodes müssen in Python existieren."""
        python_names = {code.name for code in ErrorCode}
        missing_in_python: list[str] = []

        for rust_name in rust_error_codes:
            if rust_name not in python_names:
                missing_in_python.append(rust_name)

        assert not missing_in_python, (
            f"Folgende Rust-ErrorCodes fehlen in Python:\n"
            f"{missing_in_python}\n\n"
            f"Bitte src/shared/error_codes.py aktualisieren."
        )

    def test_all_code_values_match(self, rust_error_codes: dict[str, int]) -> None:
        """Alle ErrorCode-Werte müssen zwischen Python und Rust identisch sein."""
        mismatches: list[tuple[str, int, int]] = []

        for code in ErrorCode:
            if code.name in rust_error_codes:
                rust_value = rust_error_codes[code.name]
                if code.value != rust_value:
                    mismatches.append((code.name, code.value, rust_value))

        assert not mismatches, (
            f"ErrorCode-Werte stimmen nicht überein:\n"
            + "\n".join(
                f"  {name}: Python={py_val}, Rust={rust_val}"
                for name, py_val, rust_val in mismatches
            )
            + "\n\nBitte beide Seiten synchronisieren."
        )

    def test_complete_sync_summary(self, rust_error_codes: dict[str, int]) -> None:
        """Zusammenfassender Sync-Test mit detailliertem Report."""
        python_codes = {code.name: code.value for code in ErrorCode}

        # Vergleiche Sets
        python_names = set(python_codes.keys())
        rust_names = set(rust_error_codes.keys())

        only_python = python_names - rust_names
        only_rust = rust_names - python_names
        common = python_names & rust_names

        # Prüfe Werte für gemeinsame Codes
        value_mismatches = [
            (name, python_codes[name], rust_error_codes[name])
            for name in common
            if python_codes[name] != rust_error_codes[name]
        ]

        # Generiere Report
        report_lines: list[str] = []

        if only_python:
            report_lines.append(f"Nur in Python ({len(only_python)}): {only_python}")
        if only_rust:
            report_lines.append(f"Nur in Rust ({len(only_rust)}): {only_rust}")
        if value_mismatches:
            report_lines.append(f"Wert-Mismatches ({len(value_mismatches)}):")
            for name, py_val, rust_val in value_mismatches:
                report_lines.append(f"  {name}: Python={py_val}, Rust={rust_val}")

        if report_lines:
            pytest.fail(
                "ErrorCode-Synchronisation fehlgeschlagen:\n" + "\n".join(report_lines)
            )

        # Erfolgsfall: Zeige Statistik
        print(f"\n✅ ErrorCode Sync OK: {len(common)} Codes synchronisiert")


class TestErrorCodeDocumentation:
    """Tests für ErrorCode Dokumentation."""

    def test_all_codes_have_docstrings(self) -> None:
        """Alle ErrorCodes sollten dokumentiert sein."""
        # IntEnum Members haben keine individuellen Docstrings,
        # aber wir können prüfen dass die Klasse dokumentiert ist
        assert ErrorCode.__doc__ is not None, "ErrorCode braucht einen Docstring"

    def test_error_category_covers_all_codes(self) -> None:
        """error_category muss für alle Codes funktionieren."""
        for code in ErrorCode:
            category = error_category(code)
            assert category != "UNKNOWN", f"error_category({code.name}) liefert UNKNOWN"
