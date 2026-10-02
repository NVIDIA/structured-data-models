# Spec: Geplante Target-Kategorie-Shuffles

Referenz: `tabicl==2.0.0` bei `f719c886a586ed4a29236345e319ac1ea596c478`. Baseline: `main` bei `2a73246320ea4188d8b8791e136ac8f111e112d6`.

## Problem

`ShuffleCategories` zieht auf `main` einen unabhängigen Zustand pro Estimator. TabICLv2 kombiniert Feature- und Klassen-Pattern und verwendet jedes ausgewählte Paar für `none` und `power`. Bei E=8 entstehen vier Plan-Slots: `(0,0,1,1,2,2,3,3)`. Slots können denselben gezogenen Shift enthalten und dann dedupliziert werden.

Die Semantik darf nicht aus physischen `EnsembleTable`-Positionen folgen: Das Target liegt aktuell in einer Tabelle mit acht Positionen; Layout-Inferenz würde alle acht auf einen Shift koppeln. Feste Member-IDs scheitern ebenfalls, weil E erst beim Modellaufruf bekannt ist.

## Lösung

- `RecipeExecution` erzeugt nach Bekanntwerden von E und Recipe-Alternativen einen internen Plan mit Feature-, Target- und Repräsentations-ID pro Member.
- `ShuffleCategories` verbraucht Target-IDs. Ohne Plan bleibt der allgemeine Default unabhängig; es gibt keinen TabICL-Schalter im Processor.
- Gleiche IDs teilen einen Zustand. `shift` speichert nur Offset, Kardinalität und Inverse; Missing-Codes bleiben negativ. Kompatible Targets werden gemeinsam auf CUDA berechnet.
- Modelloutputs werden vor der Reduktion memberweise auf den kanonischen Klassenraum zurückgeführt. Tensoren liegen in `BufferList`, IDs in `extra_state`; `state_dict` erhält den gefitteten Plan.

Der Processor bleibt im TabICLv2-Targetpfad. Kategoriale Features werden ausgerichtet und numerisch kodiert; dort ist kein zusätzlicher Category-Shuffle nötig.

## Ausführbares Akzeptanzbeispiel

```python
import torch
from sdm import CategoricalTensor, TableTensor
from sdm.models import TabICLv2
from sdm.processing.execution import RecipeExecution

x = TableTensor.from_tensor(torch.randn(128, 8, device="cuda"))
y = TableTensor(categorical=CategoricalTensor(
    code=torch.arange(128, device="cuda", dtype=torch.int32).remainder(5)[:, None],
    categories=(torch.arange(5, device="cuda"),),
))
members = RecipeExecution(TabICLv2.default_recipe()).fit_transform(
    x, y, None, num_members=8,
    generator=torch.Generator(device="cuda").manual_seed(7),
)
assert all(members[i].y.equal(members[i + 1].y) for i in range(0, 8, 2))
assert any(not members[i].y.equal(members[i + 2].y) for i in range(0, 6, 2))
```

Auf `main` scheitert die Paarbedingung. Der Laufzeitplan funktioniert für beliebiges E; bei ungeradem E bleibt der letzte Slot einzeln.

## GPU-Benchmark-Ergebnisse

NVIDIA L4 (23,66 GB), PyTorch `2.9.1+cu128`, CUDA 12.8, int32, eine Target-Spalte, 100 Klassen, E=8; 5 Warm-ups/30 synchronisierte Läufe. Der Prototyp misst den K=4-Worst-Case.

| Szenario                  | Main K=8 Median/p95 | Plan K=4 Median/p95 |  Peak Main → Plan |
| ------------------------- | ------------------: | ------------------: | ----------------: |
| 50.000 Zeilen, Processor  |     8,323/12,345 ms |      1,245/1,974 ms |   3,65 → 3,48 MiB |
| 500.000 Zeilen, Processor |     8,413/11,888 ms |      1,195/1,902 ms | 36,45 → 35,74 MiB |
| 500.000, reine Arithmetik |      0,255/0,308 ms |      0,250/0,390 ms | 30,52 → 15,26 MiB |

K=4 statt K=8 beschleunigt die vektorisierte Arithmetik nicht, halbiert aber deren Outputspeicher. Der **7,04×**-Gewinn des vollständigen Pfads kommt von kompakter Ausführung: 195→25 CUDA-Events, 143→25 Launches und 52→0 asynchrone Kopien (`torch.profiler`).

Feature-/Klassen-Shifts sind in TabICLv2 und TabFM dieses Repos sowie in [TabPFN](https://github.com/PriorLabs/TabPFN/blob/main/src/tabpfn/inference_config.py#L1022-L1052) belegt. Exakte Kopplung ist nur für die gepinnte TabICLv2-Referenz nachgewiesen und bleibt deshalb Modellplan, nicht Default. Nutzungshäufigkeit ist keine benchmarkbare Größe.

Reproduktion/Rohdaten: [PR #621](https://github.com/NVIDIA/structured-data-models/pull/621), `benchmark/ensemble_permutation_gpu.py` und `benchmark/results/ensemble_permutation_gpu.json`.

## Testing

- Plan-IDs/Seedverbrauch für gerade und ungerade E gegen TabICLv2 prüfen.
- Missing, 1/viele Klassen, unabhängigen Default, `shift`/`random` und Inverse vor Reduktion prüfen.
- Leeren Processor per `state_dict` laden; Outputs, IDs, Fitted-Status und Device exakt vergleichen.
