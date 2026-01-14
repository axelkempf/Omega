# Prompt: Performance-Optimierung

## Verwendung

Für: Claude Opus (Analyse), Codex-Max (Implementierung), Sonnet (Critic)

---

## Analyse-Prompt (Claude Opus)

```
Analysiere den Performance-Hotspot in <Komponente>:

## Profiling-Daten

<Flamegraph/Benchmark-Ergebnisse>

## Fragen

1. Was ist die Root Cause?
2. Welche Optimierungs-Ansätze gibt es?
3. Was sind die Trade-offs?
4. Welcher Ansatz hat das beste Verhältnis von Aufwand zu Gewinn?

Liefere: Analyse + 2-3 konkrete Vorschläge mit geschätztem Impact
```

---

## Implementierungs-Prompt (Codex-Max)

```
Implementiere Optimierung <Ansatz> für <Komponente>:

## Ansatz

<Beschreibung>

## Files

- <path/to/file>

## Constraints

- Korrektheit vor Performance
- Tests müssen weiterhin grün sein
- Determinismus erhalten
- Benchmark hinzufügen/aktualisieren

## Akzeptanzkriterien

- [ ] Performance-Verbesserung messbar (Benchmark)
- [ ] Keine Regression in Tests
- [ ] Keine Semantik-Änderung
```

---

## Critic-Prompt (Claude Sonnet)

```
Prüfe die Performance-Optimierung auf:

1. Ist die Korrektheit erhalten? (Tests)
2. Ist der Performance-Gain signifikant und messbar?
3. Gibt es versteckte Kosten (Memory, Complexity)?
4. Ist der Code noch wartbar?
5. Determinismus erhalten?

Liefere: Findings + Empfehlung
```
