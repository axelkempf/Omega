# FFI Interface Specifications

Documentation für Foreign Function Interface (FFI) Typen und Signaturen zur Vorbereitung der Rust/Julia-Migration.

## Übersicht

Dieses Verzeichnis enthält detaillierte Interface-Spezifikationen für Module, die als Migrations-Kandidaten für Rust und/oder Julia identifiziert wurden.

## Dokumentierte Module

### Phase 2 Module (Core FFI-Specs)

| Modul | Datei | Zielsprache | Task-ID | Status |
|-------|-------|-------------|---------|--------|
| `indicator_cache.py` | [indicator_cache.md](indicator_cache.md) | Rust | P2-01 | ✅ Spezifiziert |
| `event_engine.py` | [event_engine.md](event_engine.md) | Rust | P2-02 | ✅ Spezifiziert |
| `execution_simulator.py` | [execution_simulator.md](execution_simulator.md) | Rust | P2-03 | ✅ Spezifiziert |
| Rating-Module | [rating_modules.md](rating_modules.md) | Rust/Julia | P2-04 | ✅ Spezifiziert |

### Phase 6 Module (Extended FFI-Specs)

| Modul | Datei | Zielsprache | Task-ID | Status |
|-------|-------|-------------|---------|--------|
| `multi_symbol_slice.py` | [multi_symbol_slice.md](multi_symbol_slice.md) | Rust | P6-01 | ✅ Spezifiziert |
| `symbol_data_slicer.py` | [symbol_data_slicer.md](symbol_data_slicer.md) | Rust | P6-02 | ✅ Spezifiziert |
| `slippage_and_fee.py` | [slippage_fee.md](slippage_fee.md) | Rust | P6-03 | ✅ Spezifiziert |
| `portfolio.py` | [portfolio.md](portfolio.md) | Rust | P6-04 | ✅ Spezifiziert |

### Konventionen & Dokumentation

| Dokument | Datei | Status |
|----------|-------|--------|
| Nullability-Konvention | [nullability-convention.md](nullability-convention.md) | ✅ Dokumentiert |
| Data-Flow-Diagramme | [data-flow-diagrams.md](data-flow-diagrams.md) | ✅ Dokumentiert |

## ADRs (Architecture Decision Records)

| ADR | Thema | Status |
|-----|-------|--------|
| [ADR-0001](../adr/ADR-0001-migration-strategy.md) | Migrationsstrategie (Rust/Julia) | ✅ Akzeptiert |
| [ADR-0002](../adr/ADR-0002-serialization-format.md) | Serialisierungsformat (Arrow IPC) | ✅ Akzeptiert |
| [ADR-0003](../adr/ADR-0003-error-handling.md) | Fehlerbehandlungs-Konvention | ✅ Akzeptiert |

## Shared Code für FFI

| Modul | Pfad | Beschreibung |
|-------|------|--------------|
| Arrow Schemas | [src/shared/arrow_schemas.py](../../src/shared/arrow_schemas.py) | Arrow Schema-Definitionen |
| Error Codes | [src/shared/error_codes.py](../../src/shared/error_codes.py) | FFI ErrorCode Enum |
| Exceptions | [src/shared/exceptions.py](../../src/shared/exceptions.py) | Python Exception-Hierarchie |

## Konventionen

### Typ-Notation

- **Python-Typen**: Werden in Python Type Hint Syntax dargestellt
- **NumPy-DTypes**: Explizit als `np.float64`, `np.int64` etc. angegeben
- **Shape-Constraints**: Werden in Kommentaren dokumentiert: `# shape: (n,)` oder `# shape: (n, m)`
- **Nullability**: `Optional[T]` oder `T | None` für nullable Werte

### FFI-Boundary-Marker

```python
# @ffi_boundary: Input
# @ffi_boundary: Output
# @ffi_boundary: Internal (nicht über FFI exponiert)
```

### Serialisierungsformat

Primär: **Apache Arrow IPC** für numerische Daten
- Zero-Copy Transfer zwischen Python ↔ Rust
- Schema-Evolution unterstützt
- Julia-kompatibel via Arrow.jl

Fallback: **msgpack** für flexible Datenstrukturen
- Kompakter als JSON
- Schema-less

Debug: **JSON** für Konfiguration und Debugging

## Abhängigkeiten zu Phase 1

Die Interface-Spezifikationen basieren auf den in Phase 1 definierten Typen:

- `src/backtest_engine/core/types.py` - Zentrale TypedDict/TypeAlias Definitionen
- `src/shared/protocols.py` - Runtime-checkable Protocols für FFI-Boundaries

## Phase 2 Status

### Alle Tasks abgeschlossen ✅

| Task | Beschreibung | Status |
|------|--------------|--------|
| P2-01 | Input/Output-Typen für `indicator_cache.py` | ✅ Abgeschlossen |
| P2-02 | Input/Output-Typen für `event_engine.py` | ✅ Abgeschlossen |
| P2-03 | Input/Output-Typen für `execution_simulator.py` | ✅ Abgeschlossen |
| P2-04 | Input/Output-Typen für Rating-Module | ✅ Abgeschlossen |
| P2-05 | Serialisierungsformat ADR (Arrow IPC) | ✅ Abgeschlossen |
| P2-06 | Arrow-Schema-Definitionen | ✅ Abgeschlossen |
| P2-07 | Fehlerbehandlungs-Konvention ADR | ✅ Abgeschlossen |
| P2-08 | README aktualisieren | ✅ Abgeschlossen |
| P2-09 | Nullability-Konvention | ✅ Abgeschlossen |
| P2-10 | Data-Flow-Diagramme | ✅ Abgeschlossen |

### Phase 2 Completion Date: 2026-01-05

**Nächster Meilenstein:** Phase 3 - Proof-of-Concept Implementation

## Referenzen

- [Rust/Julia Migration Preparation Plan](../RUST_JULIA_MIGRATION_PREPARATION_PLAN.md)
- [ADR-0001: Migration Strategy](../adr/ADR-0001-migration-strategy.md)
- [ADR-0002: Serialization Format](../adr/ADR-0002-serialization-format.md)
- [ADR-0003: Error Handling](../adr/ADR-0003-error-handling.md)
