# Prompt: Bugfix

## Verwendung

Für: GitHub Copilot (Builder), Claude Sonnet (Critic)

---

## Prompt

```
Behebe den Bug: <Beschreibung>

## Repro

<Schritte oder Test-Case>

## Root Cause (falls bekannt)

<Hypothese>

## Files (Allowlist)

- <path/to/file>

## Constraints

- Kein Verhalten außerhalb des Bugs ändern
- Keine neuen Dependencies
- Deterministisch
- Test für den Fix schreiben

## Akzeptanzkriterien

- [ ] Bug ist behoben
- [ ] Kein Regression in bestehenden Tests
- [ ] Neuer Test dokumentiert den Fix
- [ ] Minimaler Patch (kein Refactoring)
```

---

## Critic-Prompt (Claude Sonnet)

```
Prüfe den Bugfix auf:

1. Ist der Bug wirklich behoben? (Repro-Schritt)
2. Gibt es Seiteneffekte?
3. Ist der Patch minimal?
4. Ist der Test ausreichend?
5. Determinismus gewährleistet?

Liefere: Findings + ggf. bessere Lösung
```
