# Omega V2 Context Pack

> Minimaler, aber vollständiger Kontext für V2-Implementierungsaufgaben.

## V2 Kern-Dokumente (immer verlinken bei V2-Tasks)

| Bereich | Dokument |
|---------|----------|
| **Vision** | `docs/OMEGA_V2_VISION_PLAN.md` |
| **Architektur** | `docs/OMEGA_V2_ARCHITECTURE_PLAN.md` |
| **Module/Crates** | `docs/OMEGA_V2_MODULE_STRUCTURE_PLAN.md` |
| **Execution** | `docs/OMEGA_V2_EXECUTION_MODEL_PLAN.md` |
| **Output-Contract** | `docs/OMEGA_V2_OUTPUT_CONTRACT_PLAN.md` |
| **Config-Schema** | `docs/OMEGA_V2_CONFIG_SCHEMA_PLAN.md` |
| **Metriken** | `docs/OMEGA_V2_METRICS_DEFINITION_PLAN.md` |
| **Strategien** | `docs/OMEGA_V2_STRATEGIES_PLAN.md` |
| **Indikatoren** | `docs/OMEGA_V2_INDICATOR_CACHE__PLAN.md` |
| **Trade-Mgmt** | `docs/OMEGA_V2_TRADE_MANAGER_PLAN.md` |
| **Data-Flow** | `docs/OMEGA_V2_DATA_FLOW_PLAN.md` |
| **Data-Governance** | `docs/OMEGA_V2_DATA_GOVERNANCE_PLAN.md` |
| **Testing** | `docs/OMEGA_V2_TESTING_VALIDATION_PLAN.md` |
| **CI/CD** | `docs/OMEGA_V2_CI_WORKFLOW_PLAN.md` |
| **Observability** | `docs/OMEGA_V2_OBSERVABILITY_PROFILING_PLAN.md` |
| **Tech-Stack** | `docs/OMEGA_V2_TECH_STACK_PLAN.md` |
| **Formatting** | `docs/OMEGA_V2_FORMATTING_PLAN.md` |
| **Agents** | `docs/OMEGA_V2_AGENT_INSTRUCTION_PLAN.md` |

## V2 Workspace-Struktur (Ziel)

```
rust_core/                    ← NEU: V2 Rust Workspace
├── Cargo.toml               ← Workspace-Definition
├── Cargo.lock
├── rust-toolchain.toml      ← Rust 1.76.0+
└── crates/
    ├── types/               ← Shared Types (Candle, Signal, Trade, ...)
    ├── data/                ← Data Loading (Parquet, Alignment)
    ├── indicators/          ← Indicator Engine + Cache
    ├── execution/           ← Order Execution (Slippage, Fees, Fill)
    ├── portfolio/           ← Portfolio Management (Positions, Equity)
    ├── trade_mgmt/          ← Trade Management (Rules → Actions)
    ├── strategy/            ← Strategy Interface + Implementations
    ├── backtest/            ← Event Loop + Orchestration
    ├── metrics/             ← Performance Metrics
    └── ffi/                 ← Python Binding (PyO3)
```

## Crate-Abhängigkeiten (Einweg, keine Zyklen)

```
ffi → backtest → strategy → trade_mgmt → portfolio → execution → indicators → data → types
```

## Task-spezifische Kontext-Auswahl

| Task-Typ | Pflicht-Dokumente |
|----------|-------------------|
| Neues Crate | MODULE_STRUCTURE, ARCHITECTURE, TECH_STACK |
| Execution-Logik | EXECUTION_MODEL, OUTPUT_CONTRACT, TESTING |
| Config-Änderung | CONFIG_SCHEMA, DATA_FLOW |
| Output-Änderung | OUTPUT_CONTRACT, METRICS_DEFINITION |
| Strategie | STRATEGIES, INDICATOR_CACHE, TRADE_MANAGER |
| Performance | OBSERVABILITY_PROFILING, TECH_STACK |
