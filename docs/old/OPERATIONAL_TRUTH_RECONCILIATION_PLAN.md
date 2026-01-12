# Operational Truth Reconciliation Plan (Docs ↔ System-Realität)

> Hinweis: Dieses Dokument ist **kein** ADR und **kein** Runbook. Es ist ein Arbeitsplan, um Dokumentation und Repository-Realität deckungsgleich zu machen.

## Zielbild

Die Dokumentation des Omega-Repos (Plans, ADRs, Runbooks, READMEs, Reports) soll als **operationale Wahrheit** dienen. Das bedeutet:

- Jede Behauptung in Docs ist **nachprüfbar** (Dateipfad, Test, CI-Job, Kommando, Output-Artefakt).
- Jede Referenz ist **auflösbar** (Datei existiert, Link ist korrekt, Symbole/Module stimmen).
- Jede “READY/COMPLETE”-Aussage ist durch **Gates** belegt (Tests/Checks laufen, nicht soft-fail).
- Keine “Wunschzustände” werden als Ist-Zustand markiert; solche Inhalte sind explizit als **PLANNED** gekennzeichnet.

**Nicht-Ziel:** In diesem Plan werden keine Codeänderungen implementiert. Der Plan definiert Schritte und Akzeptanzkriterien.

---

## Scope

### In Scope

- `docs/` (insb. `RUST_JULIA_MIGRATION_PREPARATION_PLAN.md`, `MIGRATION_READINESS_VALIDATION.md`, `docs/adr/*`, `docs/runbooks/*`, `docs/ffi/*`)
- Repo-weite Referenzen in `README.md`, `CONTRIBUTING.md`, `architecture.md`, `AGENTS.md`
- CI-Workflows: `.github/workflows/*.yml`
- Evidence-Artefakte: `reports/*` (Baselines, fingerprints, coverage snapshots)

### Out of Scope

- Fachliche Neudefinition der Migrationsstrategie (ADR-Änderung nur, wenn Ist/Docs-Konflikte es erzwingen).
- Re-Implementierung von Rust/Julia Modulen (außer minimalen “Import-Gates”, wenn zwingend für Wahrheit).

---

## Arbeitsprinzipien (damit es wirklich „wahr“ wird)

1. **Single Source of Truth pro Aussage**
   - Jede Aussage bekommt genau einen Beleg: (a) test, (b) CI-Step, (c) Datei/Config, (d) reproduzierbarer Befehl.

2. **Hard Gates statt Soft Claims**
   - “READY” darf nur vergeben werden, wenn entsprechende Checks **nicht** `continue-on-error` sind und lokal/CI reproduzierbar laufen.

3. **Zwei-Typen-Content**
   - **FACT**: stimmt im Repo und ist belegbar.
   - **PLANNED**: Zielzustand, klar markiert, ohne READY/COMPLETE Label.

4. **Machine-checkable References**
   - Pfade und Links werden automatisiert geprüft (z.B. via pytest oder ein kleines Docs-Lint-Skript).

---

## Inventar & „Truth Map“

Ziel: eine Tabelle (oder JSON) erzeugen, die pro Doc-Kapitel alle Claims mit Evidence verknüpft.

### Datenstruktur (minimal)

- Doc-Datei
- Abschnitt/Überschrift
- Claim (Kurztext)
- Evidence-Typ: `path|test|ci|command|artifact`
- Evidence-Ref: z.B. `tests/test_ffi_contracts.py::TestErrorCodeStability` oder `.github/workflows/ci.yml#type-check`
- Status: `OK|BROKEN_REF|STALE|PLANNED|NEEDS_GATE`
- Fix-Owner: `docs|ci|code`

**Akzeptanzkriterium:** 100% der Claims in Scope sind auf eine Evidence gemappt.

---

## Phase A – Referenz-Audit (Broken Links & Pfade)

### A1. Runbooks: Frontmatter-Links resolvable

**Warum:** Runbooks sind operational; gebrochene `rollback_procedure` Links invalidieren den Rollback.

**Schritte:**

1. Sammle alle Runbook-Frontmatter-Felder (`module`, `phase`, `prerequisites`, `rollback_procedure`).
2. Prüfe, ob `rollback_procedure` Dateien existieren.
3. Prüfe in Runbooks referenzierte Tests/Dateien (z.B. `tests/test_*.py`, `src/...`).

**Bekannte Findings (bereits verifiziert):**

- ✅ **Resolved (Phase A):** `docs/runbooks/rollback_generic.md` existiert und wird in Runbooks als `rollback_procedure` referenziert.
- ✅ **Resolved (Phase A):** `docs/runbooks/slippage_fee_migration.md` enthält keine Referenz mehr auf eine veraltete, nicht-existente Slippage/Fee-Testdatei.

**Akzeptanzkriterium:**

- 0 gebrochene Runbook-Referenzen.
- Jeder Rollback-Link zeigt auf eine existierende Datei.

**Fix-Optionen:**

- `rollback_generic.md` erstellen (generisches Rollback), oder
- Runbooks auf existierende Rollback-Doku umstellen, oder
- Frontmatter-Feld entfernen und stattdessen im Runbook explizite Rollback-Schritte inline dokumentieren.

### A2. docs/ffi: Referenzen auf tatsächliche Module und Symbole

**Schritte:**

1. Für jedes `docs/ffi/*.md`:
   - Quelle (`src/...`) existiert
   - Symbolnamen (Funktionen/Klassen) existieren oder sind als PLANNED markiert
2. Bei Abweichungen: entweder Specs aktualisieren oder Code-Skeleton hinzufügen (minimal), aber nur wenn zwingend.

**Akzeptanzkriterium:** 0 gebrochene `src/...` Pfade, 0 nicht-markierte Fantasie-Symbole.

---

## Phase B – Status-Claims harmonisieren (READY/COMPLETE vs Realität)

### B1. „Phase X ist komplett“ nur mit Gates

**Ziel:** In `docs/RUST_JULIA_MIGRATION_PREPARATION_PLAN.md`, `docs/runbooks/ready_for_migration_checklist.md`, `docs/MIGRATION_READINESS_VALIDATION.md` dürfen „✅ 100% komplett“ / „🟢 READY“ nur stehen, wenn:

- Referenzen stimmen (Phase A)
- Gates existieren und laufen (Phase C)

**Bekannte Inkonsistenzen (bereits verifiziert):**

- `MIGRATION_READINESS_VALIDATION.md` und CI-Gates müssen deckungsgleich sein: „READY“ ist nur zulässig, wenn die zugehörigen Checks **hart** failen (kein `continue-on-error`, kein `|| true`).

**Schritte:**

1. Definiere „READY“ formal (ein Satz + harte Kriterien).
2. Führe pro Phase (0–6) eine Claim-Liste mit Evidence ein.
3. Ersetze pauschale ✅-Behauptungen durch:
   - ✅ (belegt)
   - ⚠️ (teilweise, mit konkretem fehlendem Gate)
   - ⏳ PLANNED (noch nicht real)

**Akzeptanzkriterium:** Keine Docs enthalten widersprüchliche Status-Markierungen.

---

## Phase C – CI/Local Gates: Was gilt wirklich als bestanden?

### C1. Konsistenzmatrix „lokal vs CI“

**Schritte:**

1. Liste alle relevanten Gates auf:
   - `pytest` subsets (schema registry, FFI contracts, golden, property, benchmarks)
   - `mypy --strict` für migrationskritische Module
   - Rust wheel build + Import check
   - Julia package instantiate + basic import/integration check
2. Markiere pro Gate:
   - Läuft in CI hard-fail?
   - Läuft lokal reproduzierbar?
   - Ist es `continue-on-error`?

### C1.1 Gate-Matrix (Snapshot)

| Gate | CI Evidence | Hard-fail? | Lokal reproduzierbar? | Notes |
|------|------------|-----------:|-----------------------:|-------|
| Python Unit Suite | `.github/workflows/ci.yml` → job `test` | ✅ | ✅ | Läuft mit `-m "not integration"` + Coverage |
| Python Integration Suite | `.github/workflows/ci.yml` → job `integration-tests` | ✅ | ✅ | Läuft nur unter `tests/integration` |
| mypy strict (migration-critical) | `.github/workflows/ci.yml` → job `type-check` | ✅ | ✅ | `shared/` + `backtest_engine/core|config|optimizer|rating` |
| Rust wheel Import-Truth | `.github/workflows/rust-build.yml` → job `integration` | ✅ | ✅ | `python -c "import omega._rust"` nach Wheel-Install |
| Rust FFI pytest marker | `tests/test_rust_integration.py` + `rust-build.yml` | ✅ | ✅ | In CI mit `OMEGA_REQUIRE_RUST_FFI=1` (kein Skip) |
| Julia Package Tests | `.github/workflows/julia-tests.yml` → job `test` | ✅ | ✅ | `Pkg.instantiate()` + `Pkg.test()` |
| Julia FFI pytest marker | `tests/test_julia_integration.py` + `julia-tests.yml` | ✅ | ✅ | In CI mit `OMEGA_REQUIRE_JULIA_FFI=1` + `JULIA_PROJECT` |
| Cross-platform property tests | `.github/workflows/cross-platform-ci.yml` | ✅ | ✅ | Linux-only; läuft als harter Gate |
| Cross-platform hybrid integration | `.github/workflows/cross-platform-ci.yml` → job `hybrid-integration` | ✅ | ✅ | Hard gate (FFI required nur wenn Module existieren) |
| Benchmarks | `.github/workflows/benchmarks.yml` → `run-benchmarks` | ✅ | ✅ | PRs: Regressionen (>20% vs main-baseline) failen. Push main: Baseline-Vergleich aus (Bootstrap), Artefakt wird erzeugt. |

**Bekannte Findings (aktuelle Repo-Realität):**

- ✅ **Resolved:** `rust_integration` Marker ist implementiert (siehe `tests/test_rust_integration.py`) und wird in `.github/workflows/rust-build.yml` als hard gate ausgeführt.
- ✅ **Resolved:** Rust Import-Truth Gate (`import omega._rust`) ist als hard gate in `.github/workflows/rust-build.yml` vorhanden.
- ✅ **Resolved:** `julia_integration` Marker ist implementiert (siehe `tests/test_julia_integration.py`) und wird in `.github/workflows/julia-tests.yml` als hard gate ausgeführt.
- ✅ **Resolved:** Hybrid FFI Integration in `.github/workflows/cross-platform-ci.yml` ist ein hard gate (FFI wird nur erzwungen, wenn Module vorhanden sind).
- ✅ **Resolved:** Cross-platform property tests laufen als hard gate (kein `continue-on-error`).

**Akzeptanzkriterium:** „READY“ setzt voraus, dass alle Gates, die „READY“ begründen, **hard-fail** sind.

### C2. Rust Import-Truth (Packaging)

**Warum:** Ein Wheel kann bauen und dennoch zur Laufzeit nicht importierbar sein.

**Schritte:**

1. Definiere einen minimalen Gate-Test: `pip install wheel` + `python -c "import omega._rust"`.
2. Verifiziere Namenskonsistenz zwischen maturin `module-name` und PyO3 `#[pymodule]` Name.
3. Dokumentiere das Ergebnis in Docs:
   - ✅ wenn importierbar
   - ⚠️ wenn nur buildbar

**Akzeptanzkriterium:** Jeder „Rust READY“ Claim beinhaltet einen Import-Gate-Beleg.

---

## Phase D – Doku-Refactoring (Operationale Wahrheit zentralisieren)

### D1. Single Entry Point

**Problem:** Mehrere Dokumente behaupten Status (Plan, Checklist, Validation). Das lädt zu Drift ein.

**Vorschlag:**

- Definiere eine einzige Datei als „Status Source“ (z.B. `docs/MIGRATION_READINESS_VALIDATION.md`), und alle anderen referenzieren nur dorthin.
- `ready_for_migration_checklist.md` wird zu einem **Checklisten-Template**, nicht zu einer „alles ist grün“ Behauptung.

**Akzeptanzkriterium:** Es gibt genau eine Stelle, wo Status festgelegt wird.

### D2. „PLANNED“-Markierungen standardisieren

- Einheitlicher Hinweis-Block für geplante Features (z.B. Performance Targets, SIMD).
- Keine ✅ für PLANNED Tabellen.

---

## Phase E – Automatisierte Docs-Validierung (damit es so bleibt)

### E1. Docs Reference Linter (Minimal)

**Ziel:** Bei PRs darf kein neuer Broken Link / Broken Path entstehen.

**Schritte:**

1. Implementiere einen einfachen Test oder Script (pytest):
   - Findet Markdown-Referenzen auf `docs/...`, `tests/...`, `src/...`
   - Prüft Existenz
   - Optional: prüft YAML front matter Felder in Runbooks
2. In CI einhängen (Docs-Lint Job).

**Akzeptanzkriterium:** CI blockiert PRs mit gebrochenen Doc-Referenzen.

**Evidence (implementiert 2026-01-07):**

- pytest Validator: `tests/test_docs_reference_linter.py`
- CI hard gate: `.github/workflows/ci.yml` → job `docs-lint`

---

## Deliverables (konkret, nach Abschluss)

- ✅ Alle Runbooks sind selbstkonsistent (Rollback vorhanden, Referenzen korrekt).
- ✅ `RUST_JULIA_MIGRATION_PREPARATION_PLAN.md` und `MIGRATION_READINESS_VALIDATION.md` sind widerspruchsfrei.
- ✅ Eine „Truth Map“ (Tabelle/JSON) existiert.
- ✅ Ein Docs-Validator verhindert erneute Drift.

---

## Empfohlene Ausführungsreihenfolge (konservativ, risikoarm)

1. **Phase A** (Broken refs) – schnellster Wert, verhindert sofort operativen Schaden.
2. **Phase B** (Status harmonisieren) – eliminiert widersprüchliche „READY“-Claims.
3. **Phase C** (Gates) – macht „READY“ belastbar.
4. **Phase D** (Zentralisierung) – reduziert künftige Drift.
5. **Phase E** (Automatisierung) – hält die Wahrheit stabil.

---

## Definition of Done (DoD)

- Keine gebrochenen Referenzen in `docs/` (laufender Validator).
- Keine widersprüchlichen READY/COMPLETE Markierungen zwischen Plan/ADR/Validation/Checklist.
- Jeder READY Claim verweist auf mindestens einen hard-failing Gate (Test/CI).
- Rust/Julia Build/Import Truth ist explizit dokumentiert und reproduzierbar.
