# Prompt: Contract-Änderung (High-Risk)

## Verwendung

Für: Claude Opus (Builder), Claude Sonnet + GPT-5.2 (Critic)

---

## Prompt

```
Ändere den Contract für <Output/Config/Execution>:

## Änderung

<Was soll geändert werden?>

## Begründung

<Warum ist die Änderung notwendig?>

## Betroffene Dokumente

- docs/OMEGA_V2_<TOPIC>_PLAN.md

## Backward-Compatibility

<Wie wird Migration gehandhabt?>

## Akzeptanzkriterien

- [ ] Contract-Dokument aktualisiert
- [ ] Golden-Files aktualisiert
- [ ] Migration dokumentiert
- [ ] Keine stillen Semantik-Änderungen
- [ ] Tests aktualisiert und grün

## Deliverables

1. Aktualisiertes Contract-Dokument
2. Beispiel-Output (minimal)
3. Migration-Guide (wenn Breaking)
4. Aktualisierte Tests/Golden-Files
```

---

## Critic-Prompt (Sonnet + GPT-5.2)

```
Prüfe die Contract-Änderung auf:

1. Ist die Änderung wirklich notwendig?
2. Backward-Compatibility: Bricht das bestehende Konsumenten?
3. Ist die Migration klar dokumentiert?
4. Sind alle betroffenen Stellen aktualisiert?
5. Stimmen Golden-Files mit dem neuen Contract überein?
6. Gibt es versteckte Semantik-Änderungen?

Liefere: Risiko-Assessment + Findings + Empfehlung (Approve/Request Changes)
```
