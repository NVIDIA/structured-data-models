# Spec: Skalierbare Latin-Spaltenpermutationen

Referenz: `tabicl==2.0.0` bei `f719c886a586ed4a29236345e319ac1ea596c478`. Baseline: `main` bei `bb06773db1b469e027acb385d4fe43dcfe21d2ee`.

## Problem

`ShuffleColumns` ist ein allgemeiner Processor. Seine öffentliche `method` muss daher die ausgeführte Strategie beschreiben und darf sich nicht stillschweigend mit der Spaltenanzahl ändern. Die TabICLv2-Referenz verletzt dieses Prinzip: `latin` fällt oberhalb von 4.000 Spalten auf `random` zurück.

Die Grenze wurde nach einem RecursionError bei 3.000 Features in [TabICL #9](https://github.com/soda-inria/tabicl/issues/9) eingeführt. Als Begründung wurden die Kosten der eager Latin-Implementierung und die Annahme genannt, Random sei bei vielen Features ausreichend divers. Es gibt dort keine Qualitätsmessung oder Herleitung für genau 4.000. Der Referenzcode materialisiert alle `C × C` Python-Indizes; bei `C=4.000` dauert das 6,41 s und erreicht 397,7 MiB RSS. Die Grenze schützt somit diese Implementierung, nicht die Latin-Semantik.

## Allgemeine Regel und Lösung

- `random` wird der Default. Explizites `method="latin"` erzeugt bei jeder Spaltenanzahl Latin-Permutationen; explizites `random` bleibt random. Es gibt keinen impliziten Fallback.
- Aufwand und Speicher skalieren mit Spalten `C` und angeforderten Estimatoren `E`, nicht mit einem festen Grenzwert. Ein Latin-Zyklus enthält `C` balancierte Permutationen; innerhalb eines Zyklus werden IDs nicht wiederholt. Bei `E > C` beginnt ein neuer deterministisch randomisierter Zyklus.
- Ein Estimator erhält wie jeder andere genau eine Permutation der gewählten Methode. Modelle, die für `E=1` Identity benötigen, komponieren `Identity`; der generische Processor bekommt keine Modell-Sonderregel.
- Die Referenz-RNG-Aufrufe werden ohne Rekursion reproduziert. Eine Order-Statistic-Struktur zieht dieselbe Symbolfolge in `O(C log C)`; Zeilen- und Spaltenreihenfolge bleiben eindimensional. Nur die angeforderten Permutationen werden als device-lokale `torch.long`-Tensoren materialisiert. Gesamt: `O(C log C + E·C)` Zeit und `O(C + E·C)` Speicher.
- Falls später eine reale Ressourcenobergrenze belegt wird, wird sie explizit konfigurierbar oder führt zu einem klaren Fehler; sie ändert niemals unbemerkt die gewählte Methode.

Pseudo-Code:

```text
Methode und Seed festlegen
Latin-Zustand aus Symbol-, Zeilen- und Spaltenreihenfolge erzeugen
angeforderte Pattern-IDs ohne Wiederholung pro Zyklus wählen
nur diese Permutationen auf dem Eingabe-Device materialisieren
```

## Benchmark-Ergebnisse

CPU-Planung, acht ausgewählte Permutationen, sieben Läufe. Der skalierbare Prototyp stimmt für vier Seeds und `C=1…100` exakt mit der eager Referenz überein.

| Spalten |                Eager Latin | Skalierbares Latin Median/p95 |  Random Median/p95 |
| ------: | -------------------------: | ----------------------------: | -----------------: |
|     100 |           0,473 / 0,636 ms |              0,219 / 0,224 ms |                  – |
|   4.000 |   6.407 ms / 397,7 MiB RSS |              11,22 / 11,38 ms |    7,21 / 11,98 ms |
|   8.000 | nicht ausgeführt (`O(C²)`) |              31,18 / 41,57 ms |   14,89 / 20,44 ms |
|  16.000 | nicht ausgeführt (`O(C²)`) |              50,63 / 63,73 ms |   29,43 / 34,16 ms |
|  64.000 | nicht ausgeführt (`O(C²)`) |            229,68 / 233,03 ms | 123,03 / 140,24 ms |

Der vollständige skalierbare Lauf bis 64.000 Spalten erreichte 35,2 MiB Prozess-RSS. Latin ist messbar teurer als Random, aber ohne Schwelle praktisch ausführbar; der Nutzer wählt den Trade-off über `method`.

## Testing

- Bis einschließlich 4.000 Spalten Permutationen und RNG-Reproduzierbarkeit exakt mit der gepinnten Referenz vergleichen.
- Bei 4.001, 8.000 und 64.000 Spalten die Latin-Eigenschaft und das Ausbleiben eines Methodenwechsels prüfen.
- Einen/viele Estimatoren, mehrere Latin-Zyklen, `random` als Default, Inverse-Transform sowie bestehende `shift`-, CPU- und CUDA-Pfade abdecken.
