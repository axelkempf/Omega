# Omega V2 – Formatting Plan (Docs, Code, Comments)

> **Status**: Planungsphase  
> **Erstellt**: 13. Januar 2026  
> **Zweck**: Normative Regeln für Formatierung von Dokumentation, Code und Kommentaren inkl. Durchsetzung (pre-commit + CI)  
> **Referenz**: Teil der OMEGA_V2 Planungs-Suite

---

## Verwandte Dokumente

| Dokument | Fokus |
|----------|-------|
| [OMEGA_V2_VISION_PLAN.md](OMEGA_V2_VISION_PLAN.md) | Zielbild, MVP-Scope, Qualitätskriterien |
| [OMEGA_V2_ARCHITECTURE_PLAN.md](OMEGA_V2_ARCHITECTURE_PLAN.md) | Architektur, Modul-Grenzen, Single FFI Boundary |
| [OMEGA_V2_STRATEGIES_PLAN.md](OMEGA_V2_STRATEGIES_PLAN.md) | Strategie-Spezifikation (MVP: Mean Reversion Z-Score), Szenarien 1–6, Guards/Filter, Indikatoren |
| [OMEGA_V2_MODULE_STRUCTURE_PLAN.md](OMEGA_V2_MODULE_STRUCTURE_PLAN.md) | Struktur/Interfaces (wo Regeln „greifen“) |
| [OMEGA_V2_TECH_STACK_PLAN.md](OMEGA_V2_TECH_STACK_PLAN.md) | Toolchains, Version-Pinning, Build/Packaging |
| [OMEGA_V2_CI_WORKFLOW_PLAN.md](OMEGA_V2_CI_WORKFLOW_PLAN.md) | Durchsetzung/Quality Gates in CI |

---

## 1. Zusammenfassung

Dieses Dokument definiert die **universale Wahrheit** für Formatierung und Stil in Omega V2:

- **Docs**: Struktur, Lesbarkeit, konsistente Terminologie.
- **Code**: Tool-basierte Formatierung, Imports, Lint/Type Checks.
- **Kommentare**: sparsam, erklärend (WHY), ohne „dead code“.

**Normativ (MVP):**

- Formatierung ist **tool-driven** und wird **hart** durchgesetzt (lokal via `pre-commit`, in CI als Gate).
- **Hybrid Single Source of Truth**:
  - Prinzipien/Policy in diesem Plan.
  - Exakte Tool-Parameter in den Konfig-Dateien (z.B. `pyproject.toml`, `.pre-commit-config.yaml`).
- **Konflikte** zwischen Plänen/Regeln werden **nicht** still gelöst: sie erfordern ein explizites ADR.

---

## 2. Scope & Abgrenzung

### 2.1 In Scope

1. Formatierung/Style für:
  - Markdown-Dokumente (insb. `docs/OMEGA_V2_*_PLAN.md`)
  - Python-Code
  - Rust-Code (sobald V2-Core entsteht)
  - Kommentare/Docstrings
2. Durchsetzung über lokale Developer-Tools und CI.

### 2.2 Out of Scope

- **Output-Artefakte** (JSON/CSV-Schema, Rundung, Keys, Units, Pfade): siehe
  - `OMEGA_V2_OUTPUT_CONTRACT_PLAN.md`
  - `OMEGA_V2_METRICS_DEFINITION_PLAN.md`
- Umfassende Naming-Konventionen für Runtime-Pfade/Artefaktordner: nicht Teil dieses Plans (operational kritisch, siehe `var/`-Guardrails in Repo-Doku).

---

## 3. Durchsetzung (Gates)

### 3.1 Lokal (pre-commit)

**Normativ (MUSS):**

- `pre-commit` ist für Contributors verpflichtend.
- Reformatting erfolgt über Hooks (kein „Hand-Flicken“).

### 3.2 CI

**Normativ (MUSS):**

- CI führt dieselben Quality Gates aus.
- Ein PR ist nur mergebar, wenn die Gates grün sind.

Referenz: `OMEGA_V2_CI_WORKFLOW_PLAN.md`.

---

## 4. Tooling-Regeln (Prinzipien, Parameter in Config)

### 4.1 Python

**Normativ (MUSS):**

- Python-Version: $\ge 3.12$ (Repo-Policy).
- Formatter: **Black**.
- Imports: **isort** mit `profile = "black"`.
- Lint: **flake8**.
- Typen: **mypy** (mindestens für öffentliche APIs; schrittweise strengere Abdeckung erlaubt/erwünscht).
- Security: **bandit**.
- Docstrings: **pydocstyle** (Google-Konvention).

**Hinweis (Hybrid Truth):**
Die exakten Parameter (z.B. line-length, ignores, excludes) stehen in:

- `pyproject.toml`
- `.pre-commit-config.yaml`

### 4.2 Rust (Omega V2 Core)

**Normativ (MUSS):**

- Formatter: `cargo fmt`.
- Lint: `cargo clippy -D warnings`.

Details (Workspace-/Crate-Struktur) siehe `OMEGA_V2_MODULE_STRUCTURE_PLAN.md`.

### 4.3 Markdown / Plan-Dokumente

**Normativ (MUSS):**

- Jede Plan-Datei enthält einen Abschnitt `## Verwandte Dokumente` mit konsistenter Link-Tabelle.
- Links sind relativ (innerhalb `docs/`).
- Keine „Dekorations-Kommentare“/Trennlinien-Orgien; Struktur erfolgt über Überschriften.

**SOLL (Konvention):**

- Überschriften-Hierarchie sauber (H1 → H2 → H3).
- Zeilenlängen pragmatisch halten (Lesbarkeit), ohne dogmatische 80-Zeichen-Regel.

---

## 5. Kommentar-Policy

**Normativ (MUSS):**

- Kommentare erklären **WHY**, nicht **WHAT**.
- Kein auskommentierter Code (kein "dead code").
- TODOs nur, wenn sie eine klare nächste Aktion beschreiben (idealerweise mit Referenz).

---

## 6. Konflikt-Handling (ADR-Pflicht)

Wenn Regeln/Pläne widersprüchlich werden (z.B. Line-Length vs. Tooling, oder Doc-Template vs. bestehender Standard):

- Es wird ein **ADR** erstellt.
- Das ADR benennt Konflikt, Entscheidung, Begründung und Migrationsschritte.
- Alle betroffenen Pläne verlinken das ADR im Abschnitt „Verwandte Dokumente“ oder an der relevanten Stelle.

---

## 7. Akzeptanzkriterien (Definition of Done)

1. `pre-commit` läuft lokal erfolgreich (alle Hooks grün).
2. CI-Gates sind grün.
3. Alle `docs/OMEGA_V2_*_PLAN.md` verlinken diesen Plan unter `## Verwandte Dokumente`.
4. Keine inkonsistenten „doppelten Wahrheiten“: Tool-Parameter sind nicht in mehrere Dokumente kopiert.
