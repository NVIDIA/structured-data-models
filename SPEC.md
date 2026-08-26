# Spec: Geplante Spalten-Shuffles auf gestapelten Tabellen

Referenz: `tabicl==2.0.0` bei `f719c886a586ed4a29236345e319ac1ea596c478`. Baseline: `main` bei `d2d89895b0540c01b1213ea1accf391db7fb74cd`.

## Problem

Eine `EnsembleTable` trennt logische Estimatoren von physisch gespeicherten Tabellen. Im aktuellen TabICLv2-Feature-Rezept erzeugt die `none`-/`power`-`Choice` acht logische Estimatoren, aber nur zwei kompatible Tabellen in einer Gruppe `[2, Zeilen, Spalten]`. Die Referenz wählt vier Shuffle-Konfigurationen und wendet jede auf beide Normalisierungen an.

`ShuffleColumns` verarbeitet auf `main` trotzdem acht logische Mitglieder einzeln. Danach hält es acht temporäre Tabellen, bevor `from_tables()` dieselben Normalisierungspaare wieder stapelt. Schema-Gleichheit allein erlaubt jedoch kein Teilen: Nur dieselbe geplante Permutations-ID darf gemeinsam ausgeführt werden; unterschiedliche Spaltenzahlen oder inkompatible Tabellen müssen getrennt bleiben.

## Lösung

- Der gemeinsame logische Plan aus #619 wird pro tatsächlich vorkommendem Schema materialisiert: nicht vorhandene Spalten werden vor der lokalen Latin-Konstruktion herausgefiltert. Jede logische Estimator-ID verweist anschließend auf eine schemaspezifische Permutations-ID; ein konkreter `[P, C]`-Tensor wird nie zwischen verschiedenen Spaltenzahlen geteilt.
- Die konkreten Permutationstensoren werden pro Schema in `BufferList` registriert. Permutations-, Schema- und Member-IDs liegen als `extra_state` vor, sodass der vollständige Ausführungsplan über den normalen `state_dict` gespeichert, geladen und auf ein anderes Device verschoben werden kann.
- Die Eingaben werden nach kompatibler gespeicherter Gruppe und Spaltenzahl partitioniert. Für eine Partition werden die eindeutigen Indizes zu `[P, C]` gestapelt und alle gespeicherten Tabellen mit einem gebündelten CUDA-`gather` verarbeitet.
- Die Quelltabelle `[S, R, C]` wird nur als View auf `[P, S, R, C]` erweitert; `gather` materialisiert ausschließlich die benötigte Ausgabe. Für TabICLv2 entsteht so ein Tensor `[4, 2, R, C]` statt acht Einzelresultaten plus Restacking.
- Ein schmaler interner `EnsembleTable`-Assembly-Pfad übernimmt die fertigen Gruppen und `(group, position)`-Locations direkt. Er erhält logische Reihenfolge und Sharing, führt keine erneute Tensor-Kopie aus und erzeugt keine neue öffentliche Container-Abstraktion.
- Inkompatible Gruppen werden separat gebatcht. Wenn kein gemeinsames Batch möglich ist, bleibt der bestehende per-Member-Pfad der korrekte Fallback.

```text
einen gemeinsamen logischen Plan für alle Gruppen lesen
pro Schema fehlende Spalten entfernen und den lokalen [P,C]-Tensor aus BufferList wählen
kompatible gespeicherte Tabellen zu [S,R,C] zusammenfassen
einmal gather auf [P,S,R,C] ausführen
Ausgabegruppen und logische Locations direkt zusammensetzen
```

## GPU-Benchmark-Ergebnisse

NVIDIA L4, PyTorch 2.13/CUDA 13, float32, 100 Spalten, acht Estimatoren, zwei gespeicherte Tabellen und vier Permutationen; synchronisierte Median/p95-Wall-Time über 30 Läufe.

| Zeilen | Main: einzeln + Restack | Vier `index_select` ohne Restack | Ein batched `gather` | Peak Main → batched |
| -----: | ----------------------: | -------------------------------: | -------------------: | ------------------: |
|  1.000 |         7,922/12,597 ms |                   2,396/2,791 ms |       0,718/1,191 ms |     6,12 → 3,06 MiB |
| 10.000 |          7,666/8,775 ms |                   1,645/2,099 ms |       0,727/0,992 ms |   63,82 → 30,52 MiB |
| 50.000 |          8,403/8,937 ms |                   1,642/2,165 ms |       1,529/1,564 ms | 312,60 → 152,59 MiB |

Bei 50.000 Zeilen war batched `gather` auch für 1/2/4/8 eindeutige Permutationen schneller als getrenntes `index_select` (0,451/0,808/1,536/2,977 ms gegenüber 0,478/0,956/1,725/3,203 ms). Der CUDA-Profiler reduziert 64 auf 6 Kernel-Events pro Anwendung. Damit ersetzt die Evidenz den bisherigen Vorschlag „mehrfach `index_select`“ durch den gebündelten Algorithmus; ein datenabhängiger Schwellwert ist nicht nötig.

## Teststrategie

- Daten, Spaltennamen, inverse Transformation und logische Reihenfolge für geteilte Tabellen, zwei Normalisierungen, wiederholte IDs und inkompatible Schemata gegen den bestehenden Pfad vergleichen.
- Die öffentliche Ausgabe für eine, zwei, vier und acht Permutationen sowie CPU-Fallback und CUDA-Pfad prüfen, ohne private Helper-Struktur festzuschreiben.
- Unterschiedliche, stark überlappende Schemata prüfen: gemeinsame Spalten behalten die gekoppelte Planung, jedes lokale Ergebnis bleibt eine gültige Permutation und keine falsche Tensorlänge wird wiederverwendet.
- Nach Fit den `state_dict` in einen leeren Processor laden und Transformation, Inverse, Gruppen-/Member-Zuordnung, Fitted-Status und Device exakt vergleichen.
- 1.000, 10.000 und 50.000 Zeilen als permanente synchronisierte GPU-Nichtregressionsszenarien behalten; Peak-Speicher zusätzlich für die große Tabelle messen.
