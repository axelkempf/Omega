"""
Golden-File Test Framework für Omega Trading System.

P3-08: Golden-File Infrastruktur für Determinismus-Validierung.

Dieses Modul bietet Tools zum Generieren, Speichern und Vergleichen von
Referenz-Outputs für Backtests und Optimizer-Runs. Es dient als kritische
Validierung für die FFI-Migration zu Rust/Julia.

Struktur:
- conftest.py: pytest Fixtures und Golden-File Utilities
- test_golden_backtest.py: P3-09 - Backtest-Determinismus
- test_golden_optimizer.py: P3-10 - Optimizer-Determinismus
- reference/: Gespeicherte Referenz-Outputs (JSON/CSV)
"""
