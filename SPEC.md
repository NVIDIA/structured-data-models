# Spec: Skalierbare Latin-Spaltenpermutationen

Referenz: `tabicl==2.0.0` bei `f719c886a586ed4a29236345e319ac1ea596c478`. Baseline: `main` bei `bebaef1ccd074a376dd2f4c3e5c0c010fb8d6db3`.

## Problem

`ShuffleColumns` ist ein allgemeiner Processor. Seine gewählte Methode darf sich deshalb nicht unbemerkt mit der Spaltenanzahl ändern. TabICLv2 wechselt bei mehr als 4.000 Spalten von `latin` zu `random`; laut [TabICL #9](https://github.com/soda-inria/tabicl/issues/9) schützt diese Grenze die rekursive `O(C²)`-Implementierung. Für genau 4.000 gibt es weder eine Qualitätsmessung noch eine allgemeine algorithmische Begründung.

Die Referenz erzeugt eine zufällige Symbol-, Zeilen- und Spaltenreihenfolge eines zyklischen Latin-Quadrats. Diese Verteilung lässt sich auf CUDA ohne das vollständige Quadrat erzeugen. Für identische TabICLv2-Seeds muss zusätzlich der Python-RNG-Verbrauch exakt reproduziert werden; diese teure Anforderung gehört in den Modellplan, nicht als Sonderregel in den allgemeinen Processor. Auf aktuellem `main` wird dynamisch großer Processor-Zustand mit `BufferList` gespeichert und über den normalen PyTorch-`state_dict` wiederhergestellt; der neue Plan muss diesen Vertrag verwenden.

## Lösung

- `random` wird der Default. Explizites `method="latin"` bleibt bei jeder Spaltenanzahl Latin; ein Ressourcenlimit meldet einen Fehler oder ist explizit konfigurierbar, ändert aber nie die Methode.
- Ein gemeinsamer logischer Plan vergibt mit einem Seed stabile Zufallsränge für die Vereinigungsmenge aller Spalten und Pattern. Für jedes tatsächlich vorkommende Schema werden nicht vorhandene Spalten aus diesen Rangfolgen entfernt, die übrigen lokal neu nummeriert und daraus ein gültiger Latin-Plan erzeugt. Stark überlappende Gruppen bleiben so gekoppelt; konkrete Tensoren bleiben wegen unterschiedlicher Spaltenzahlen schemaspezifisch.
- Der allgemeine CUDA-Pfad materialisiert pro Schema nur die benötigten `E` Permutationen als Tensor `[E, C]`: `base[(pattern[:, None] - rows[None, :]) % C]`.
- Der interne TabICLv2-Estimatorplan bildet für exakte Seed-Parität dieselben Python-RNG-Ziehungen ab. Eine Order-Statistic-Struktur dekodiert die Symbolfolge iterativ in `O(C log C)`; anschließend werden nur die ausgewählten Pattern-IDs auf CUDA materialisiert. Der generische Processor erhält fertige IDs/Zustände und kennt keine TabICLv2-Schwelle.
- Die konkreten Permutationstensoren werden in `BufferList` registriert; Schema- und Estimatorzuordnungen liegen als nicht-tensorieller `extra_state` vor. Dadurch stellt `load_state_dict()` einen gefitteten Processor ohne erneutes Fitten vollständig wieder her.

## Ausführbares Akzeptanzbeispiel

Dieser Draft ist noch spec-only. Der folgende öffentliche CUDA-Test ist das Abnahmekriterium für die spätere Implementierung.

```python
import torch
import sdm
from sdm.processing import ShuffleColumns
from sdm.tensor import EnsembleTable

device, columns = torch.device("cuda"), ("a", "b", "c", "d")
table = sdm.TableTensor.from_tensor(
    torch.arange(8, dtype=torch.float32, device=device).reshape(2, 4),
    columns=columns,
)
ensemble = EnsembleTable(table, num_members=4)
processor = ShuffleColumns(method="latin")
output = processor.fit_transform_ensemble(
    ensemble, generator=torch.Generator(device=device).manual_seed(7)
)
orders = [output.table(i).columns[sdm.Stype.numerical] for i in range(4)]
assert all({order[i] for order in orders} == set(columns) for i in range(4))
restored = ShuffleColumns(method="latin")
restored.load_state_dict(processor.state_dict())
again = restored.transform_ensemble(ensemble)
assert all(output.table(i).equal(again.table(i)) for i in range(4))
print("latin=True, state_dict=True")
```

Auf aktuellem `main` und diesem spec-only Draft endet der Lauf mit `AssertionError`, weil `latin` noch fehlt. Nach der Implementierung lautet die Ausgabe `latin=True, state_dict=True`.

## GPU-Benchmark-Ergebnisse

NVIDIA L4, PyTorch 2.13/CUDA 13, acht Patterns, synchronisierte End-to-End-Wall-Time nach Warm-up; Median/p95 über 10–20 Läufe. Der exakte kompakte Plan stimmt für vier Seeds und `C=1…100` elementweise mit der gepinnten Referenz überein.

| Spalten | Exakt: Host-Materialisierung | Exakt: CUDA-Materialisierung | Allgemeines CUDA-Latin | Peak exakt auf CUDA |
| ------: | ---------------------------: | ---------------------------: | ---------------------: | ------------------: |
|     100 |               0,307/0,330 ms |               0,377/0,568 ms |         0,236/0,289 ms |            0,02 MiB |
|   4.000 |             11,771/12,315 ms |             11,553/12,042 ms |         0,335/0,470 ms |            0,55 MiB |
|  16.000 |             53,036/58,404 ms |             52,462/65,316 ms |         0,347/0,388 ms |            2,20 MiB |
|  64.000 |           251,510/260,869 ms |           244,946/249,433 ms |         0,352/0,392 ms |            8,79 MiB |

Der Profiler bestätigt die Entscheidung: Bei `C=64.000` sinkt die reine Materialisierung von 6,09 ms auf 0,097 ms; der verbleibende exakte Aufwand kommt fast vollständig vom seriellen Python-RNG-Plan. Das CUDA-Latin ist rund 696-mal schneller als der exakte Lauf, ist aber wegen anderer Seed-Samples kein Ersatz für TabICLv2-Ausführungsparität.

## Teststrategie

- Verteilungseigenschaften und Latin-Invarianten des allgemeinen CUDA-Pfads für kleine/große `C`, mehrere `E`, mehrere Zyklen und deterministische `torch.Generator` prüfen.
- Den internen TabICLv2-Plan bis 4.000 Spalten und über mehrere Seeds exakt vergleichen; oberhalb davon Latin-Invarianten und fehlenden Methodenwechsel prüfen.
- Überlappende und unterschiedliche Gruppenschemata sowie inverse Transformation und vollständige per-Estimator-Modellinputs abdecken.
- Nach Fit `state_dict` in einen leeren Processor laden und Transformation, Inverse, Fitted-Status, Member-Zuordnung und Device exakt vergleichen.
