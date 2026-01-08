# -*- coding: utf-8 -*-
"""
Benchmark Suite für Migrations-Kandidaten (Phase 3: Test-Infrastruktur).

Dieses Paket enthält pytest-benchmark Tests für:
- indicator_cache.py: Indikator-Berechnungen (EMA, RSI, MACD, etc.)
- event_engine.py: Event-Loop Throughput und Latenz
- rating/*.py: Score-Berechnungen

Verwendung:
    # Alle Benchmarks ausführen
    pytest tests/benchmarks/ -v

    # Mit JSON-Export
    pytest tests/benchmarks/ --benchmark-json=var/results/benchmarks/latest.json

    # Nur bestimmte Benchmarks
    pytest tests/benchmarks/test_bench_indicator_cache.py -v

    # Benchmark-Vergleich gegen Baseline
    pytest tests/benchmarks/ --benchmark-compare=var/results/benchmarks/baseline.json

Siehe auch:
- docs/RUST_JULIA_MIGRATION_PREPARATION_PLAN.md (Phase 3)
- reports/performance_baselines/README.md
"""
