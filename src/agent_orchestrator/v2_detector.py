"""V2 Backtest-Core Context Detection.

Erkennt automatisch ob ein Task/Request den V2 Backtest-Core (Rust + Python Wrapper)
betrifft basierend auf Pfaden, Schlüsselwörtern und Patterns.

Siehe: OMEGA_V2_ARCHITECTURE_PLAN.md, omega-v2-backtest.instructions.md
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

# V2-spezifische Pfad-Patterns
V2_PATH_PATTERNS: Final[list[str]] = [
    r"rust_core/",
    r"python/bt/",
    r"crates/",
    r"Cargo\.(toml|lock)",
    r"\.rs$",
    r"_native\.pyi?$",
]

# V2-spezifische Schlüsselwörter im Text
V2_KEYWORDS: Final[list[str]] = [
    "rust_core",
    "python/bt",
    "maturin",
    "pyo3",
    "pyfunction",
    "pymodule",
    "golden-file",
    "golden_file",
    "golden file",
    "v2 backtest",
    "v2-backtest",
    "backtest-core",
    "ffi boundary",
    "ffi-boundary",
    "single ffi",
    "run_backtest",
    "omega_bt",
    "v1 parity",
    "v1-parity",
    "v1↔v2",
    "determinismus",
    "rng_seed",
]

# V2 Rust Crates
V2_CRATES: Final[list[str]] = [
    "types",
    "data",
    "indicators",
    "execution",
    "portfolio",
    "trade_mgmt",
    "strategy",
    "backtest",
    "metrics",
    "ffi",
]


@dataclass
class V2DetectionResult:
    """Ergebnis der V2-Kontext-Erkennung."""

    is_v2: bool
    confidence: float  # 0.0 - 1.0
    matched_patterns: list[str] = field(default_factory=list)
    matched_keywords: list[str] = field(default_factory=list)
    affected_crates: list[str] = field(default_factory=list)
    involves_ffi: bool = False
    involves_golden: bool = False
    involves_parity: bool = False


class V2Detector:
    """Erkennt V2 Backtest-Core Kontext.

    Der V2 Backtest-Core ist die Rust-basierte Neuimplementierung des Backtesting-Systems.
    Er unterscheidet sich fundamental von V1 (Python-only) durch:
    - Single FFI Boundary (run_backtest als einziger Entry-Point)
    - Rust Crate-Struktur unter rust_core/crates/
    - Python Wrapper unter python/bt/
    - Golden-File Testing für Determinismus/Parität

    Diese Klasse hilft dem Orchestrator, Tasks automatisch dem richtigen Context
    zuzuordnen und die entsprechenden Instructions zu laden.
    """

    def __init__(self) -> None:
        """Initialisiere den V2 Detector mit kompilierten Patterns."""
        self._path_patterns = [re.compile(p) for p in V2_PATH_PATTERNS]
        self._keyword_patterns = [
            re.compile(re.escape(kw), re.IGNORECASE) for kw in V2_KEYWORDS
        ]

    def detect(self, text: str, paths: list[str] | None = None) -> V2DetectionResult:
        """Analysiere Text und Pfade auf V2-Kontext.

        Args:
            text: Text zu analysieren (Task-Beschreibung, Code, etc.)
            paths: Optional - Liste von Dateipfaden

        Returns:
            V2DetectionResult mit Details zur Erkennung
        """
        matched_patterns: list[str] = []
        matched_keywords: list[str] = []
        affected_crates: list[str] = []

        # Pfad-Pattern Matching
        if paths:
            for path in paths:
                for pattern in self._path_patterns:
                    if pattern.search(path):
                        matched_patterns.append(f"{pattern.pattern} -> {path}")
                        break

        # Keyword Matching im Text
        text_lower = text.lower()
        for i, pattern in enumerate(self._keyword_patterns):
            if pattern.search(text):
                matched_keywords.append(V2_KEYWORDS[i])

        # Pfad-Patterns auch im Text suchen
        for pattern in self._path_patterns:
            if pattern.search(text):
                matched_patterns.append(f"{pattern.pattern} (in text)")

        # Betroffene Crates erkennen
        for crate in V2_CRATES:
            # Suche nach crate-Namen in Pfaden und Text
            crate_pattern = rf"crates/{crate}[/\s]|{crate}::|{crate}\s+crate"
            if re.search(crate_pattern, text, re.IGNORECASE):
                affected_crates.append(crate)
            if paths:
                for path in paths:
                    if f"crates/{crate}" in path:
                        if crate not in affected_crates:
                            affected_crates.append(crate)

        # Spezifische Feature-Erkennung
        involves_ffi = self._check_ffi_involvement(text, paths, affected_crates)
        involves_golden = self._check_golden_involvement(text, paths)
        involves_parity = self._check_parity_involvement(text)

        # Confidence berechnen
        confidence = self._calculate_confidence(
            matched_patterns,
            matched_keywords,
            affected_crates,
            involves_ffi,
            involves_golden,
            involves_parity,
        )

        is_v2 = confidence >= 0.3  # Threshold für V2-Klassifikation

        return V2DetectionResult(
            is_v2=is_v2,
            confidence=confidence,
            matched_patterns=matched_patterns,
            matched_keywords=matched_keywords,
            affected_crates=affected_crates,
            involves_ffi=involves_ffi,
            involves_golden=involves_golden,
            involves_parity=involves_parity,
        )

    def is_v2_context(self, text: str, paths: list[str] | None = None) -> bool:
        """Schnelle Prüfung ob V2-Kontext vorliegt.

        Args:
            text: Text zu analysieren
            paths: Optional - Liste von Dateipfaden

        Returns:
            True wenn V2-Kontext erkannt wurde
        """
        return self.detect(text, paths).is_v2

    def get_affected_crates(
        self, text: str, paths: list[str] | None = None
    ) -> list[str]:
        """Ermittle welche V2 Crates betroffen sind.

        Args:
            text: Text zu analysieren
            paths: Optional - Liste von Dateipfaden

        Returns:
            Liste der betroffenen Crate-Namen
        """
        return self.detect(text, paths).affected_crates

    def involves_ffi(self, text: str, paths: list[str] | None = None) -> bool:
        """Prüfe ob FFI-Grenze betroffen ist.

        Die FFI-Grenze ist besonders kritisch, da sie der einzige
        Kommunikationspunkt zwischen Python und Rust ist.

        Args:
            text: Text zu analysieren
            paths: Optional - Liste von Dateipfaden

        Returns:
            True wenn FFI-Boundary betroffen ist
        """
        return self.detect(text, paths).involves_ffi

    def involves_golden(self, text: str, paths: list[str] | None = None) -> bool:
        """Prüfe ob Golden-File Testing betroffen ist.

        Golden-Files sind kritisch für Determinismus und V1↔V2 Parität.

        Args:
            text: Text zu analysieren
            paths: Optional - Liste von Dateipfaden

        Returns:
            True wenn Golden-File Testing betroffen ist
        """
        return self.detect(text, paths).involves_golden

    def _check_ffi_involvement(
        self, text: str, paths: list[str] | None, crates: list[str]
    ) -> bool:
        """Prüfe ob FFI-Grenze involviert ist."""
        ffi_indicators = [
            "ffi",
            "pyo3",
            "pyfunction",
            "pymodule",
            "run_backtest",
            "_native",
            "omega_bt",
            "maturin",
        ]

        text_lower = text.lower()
        if any(ind in text_lower for ind in ffi_indicators):
            return True

        if "ffi" in crates:
            return True

        if paths:
            for path in paths:
                if "ffi" in path or "_native" in path:
                    return True

        return False

    def _check_golden_involvement(self, text: str, paths: list[str] | None) -> bool:
        """Prüfe ob Golden-File Testing involviert ist."""
        golden_indicators = [
            "golden",
            "expected/",
            "fixtures/",
            "test_golden",
            "update-golden",
            "--update-golden",
        ]

        text_lower = text.lower()
        if any(ind in text_lower for ind in golden_indicators):
            return True

        if paths:
            for path in paths:
                if "golden" in path.lower() or "expected" in path or "fixtures" in path:
                    return True

        return False

    def _check_parity_involvement(self, text: str) -> bool:
        """Prüfe ob V1↔V2 Parität involviert ist."""
        parity_indicators = [
            "parity",
            "parität",
            "v1↔v2",
            "v1<->v2",
            "v1 vs v2",
            "v1_parity",
            "execution_variant",
            "kanonische szenarien",
            "canonical scenarios",
        ]

        text_lower = text.lower()
        return any(ind in text_lower for ind in parity_indicators)

    def _calculate_confidence(
        self,
        patterns: list[str],
        keywords: list[str],
        crates: list[str],
        ffi: bool,
        golden: bool,
        parity: bool,
    ) -> float:
        """Berechne Konfidenz-Score für V2-Klassifikation.

        Gewichtung:
        - Pfad-Patterns: 0.3 pro Match (max 0.6)
        - Keywords: 0.15 pro Match (max 0.45)
        - Crates: 0.1 pro Match (max 0.3)
        - FFI: +0.2
        - Golden: +0.15
        - Parity: +0.15
        """
        score = 0.0

        # Pfad-Patterns (stark gewichtet)
        score += min(len(patterns) * 0.3, 0.6)

        # Keywords
        score += min(len(keywords) * 0.15, 0.45)

        # Crates
        score += min(len(crates) * 0.1, 0.3)

        # Feature-Flags
        if ffi:
            score += 0.2
        if golden:
            score += 0.15
        if parity:
            score += 0.15

        return min(score, 1.0)

    def get_v2_instructions(self) -> list[str]:
        """Gib die relevanten V2 Instruction-Pfade zurück.

        Returns:
            Liste der V2-spezifischen Instruction-Dateien
        """
        return [
            ".github/instructions/omega-v2-backtest.instructions.md",
            ".github/instructions/ffi-boundaries.instructions.md",
            ".github/instructions/rust.instructions.md",
        ]

    def get_v2_plan_docs(self) -> list[str]:
        """Gib die relevanten V2 Planungsdokumente zurück.

        Returns:
            Liste der V2-spezifischen Plan-Dokumente
        """
        return [
            "docs/OMEGA_V2_ARCHITECTURE_PLAN.md",
            "docs/OMEGA_V2_TECH_STACK_PLAN.md",
            "docs/OMEGA_V2_TESTING_VALIDATION_PLAN.md",
            "docs/OMEGA_V2_OUTPUT_CONTRACT_PLAN.md",
            "docs/OMEGA_V2_CI_WORKFLOW_PLAN.md",
        ]
