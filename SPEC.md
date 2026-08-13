# Spec: Estimatorgekoppelte Target-Kategorie-Shuffles

Referenz: `tabicl==2.0.0` bei `f719c886a586ed4a29236345e319ac1ea596c478`. Baseline: `main` bei `bb06773db1b469e027acb385d4fe43dcfe21d2ee`.

## Problem

`ShuffleCategories` arbeitet auf `main` bereits estimatorabhängig, zieht aber für jeden logischen Estimator unabhängig eine Permutation. TabICLv2 bildet zuerst Paare aus Feature- und Klassen-Pattern, mischt diese mit einem Seed und führt jedes ausgewählte Paar einmal mit `none`- und einmal mit `power`-Normalisierung aus. Beide Normalisierungen müssen deshalb denselben Klassen-Shift verwenden.

Der Processor gehört im TabICLv2-Rezept nur in den Target-Pfad. Kategoriale Features werden ausgerichtet, numerisch kodiert und später durch `ShuffleColumns` abgedeckt. Ein zusätzlicher Feature-Category-Shuffle würde Repräsentationen vervielfachen, ohne Referenzverhalten herzustellen. Vor der Estimatorreduktion muss außerdem jede Modell-Ausgabe wieder dieselbe kanonische, nach Werten sortierte Klassenachse beschreiben.

## Lösung

- Ein privater, modellseitiger Estimatorplan ordnet jeder globalen Estimator-ID `(feature_pattern_id, class_shift_id, normalization_id)` zu. Er erzeugt die Feature-/Klassen-Kombination und Seed-Mischung exakt wie die Referenz; die Normalisierungsduplizierung geschieht erst danach.
- `ShuffleCategories` erhält einen schmalen internen Resolver für vorgeplante Permutations-IDs. Ohne Plan bleibt das allgemeine unabhängige Verhalten unverändert; es entsteht kein öffentlicher TabICLv2-Modus.
- Für `method="shift"` speichert der allgemeine Zustand nur den Offset statt eines materialisierten Mappings der Länge `K`. Gleiche Offsets werden einmal gehalten und von allen zugehörigen Estimatoren referenziert.
- Kompatible Target-Tabellen werden für alle eindeutigen Offsets auf CUDA in einem Durchlauf berechnet: `(code[None] - shifts[:, None, None]) % K`. Ein `where` erhält negative Missing-Codes; die invers verschobenen Kategorie-Vektoren sichern dieselben dekodierten Werte. `random` nutzt weiterhin das allgemeine Mapping.
- Nach dem Modell wird jede Estimatorausgabe mit ihrem inversen Shift auf den kanonischen Klassenraum abgebildet und erst dann reduziert.

```text
kanonisches Target fitten -> gekoppelte Estimator-IDs planen -> eindeutige Shifts gebündelt anwenden
Modell pro Estimator -> Klassenachse invers kanonisieren -> Estimatoren reduzieren
```

## GPU-Benchmark-Ergebnisse

NVIDIA L4, PyTorch 2.13/CUDA 13, int32-Codes, eine Target-Spalte, 10 % Missing, acht Estimatoren und vier gekoppelte Shifts; synchronisierte Median/p95-Wall-Time über 30 Läufe.

| Zeilen / Klassen | Main: 8 unabhängig | Nur gekoppelt: 4 Mappings | Gekoppelte Shift-Arithmetik | Peak Main → Arithmetik |
| ---------------: | -----------------: | ------------------------: | --------------------------: | ---------------------: |
|      50.000 / 10 |     6,005/7,480 ms |            4,288/6,471 ms |              0,728/0,814 ms |        2,44 → 1,57 MiB |
|     50.000 / 100 |     6,107/6,407 ms |            3,176/3,652 ms |              0,766/0,986 ms |        2,44 → 1,57 MiB |
|    500.000 / 100 |     6,161/6,836 ms |            3,162/3,414 ms |              0,735/0,832 ms |      26,37 → 15,74 MiB |

Eine Lookup-Tabelle war bei 50.000 Zeilen/100 Klassen mit 0,800 ms langsamer und brauchte 2,34 MiB. Der CUDA-Profiler zählt pro Anwendung 192 Kernel-Events auf `main`, aber nur 15 für die Shift-Arithmetik. Damit ist nicht nur das Koppeln korrekt: Die kompakte Arithmetik ist unter den semantisch gleichwertigen Kandidaten auch der schnellste gemessene Pfad.

## Teststrategie

- Feature-/Klassen-Plan, globale IDs und Seed-Verbrauch für Klassifikation und Regression exakt mit der gepinnten Referenz vergleichen.
- Gleiche `none`-/`power`-Shifts, Missing-Codes, eine/viele Klassen, deterministische Seeds sowie vektorisierte und sequenzielle Estimatorausführung prüfen.
- Shift-Arithmetik elementweise gegen das bestehende Mapping vergleichen; `random` und unabhängiges generisches Shuffling unverändert testen.
- Per-Estimator-Ausgaben nach kanonischer Rückabbildung, Reduktion, finale Klassenreihenfolge und öffentliche Wahrscheinlichkeiten vergleichen.
