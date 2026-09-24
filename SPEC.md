# Spec: Planbasierte, gebatchte Spalten-Shuffles

Referenz: `tabicl==2.0.0` bei `f719c886a586ed4a29236345e319ac1ea596c478`. Baseline: `main` bei `2a73246320ea4188d8b8791e136ac8f111e112d6`.

## Problem

Nach `Choice(Identity, PowerTransform)` enthält die `EnsembleTable` zwei Datenrepräsentationen. TabICLv2 wendet jede ausgewählte Shuffle-Konfiguration auf beide an. Bei E=8 sind das vier Slots und höchstens vier eindeutige Spaltenpermutationen. `ShuffleColumns` auf `main` zieht und verarbeitet dagegen acht Permutationen einzeln.

Das Modell entscheidet, **welche** Member eine logische Permutation teilen; der Processor entscheidet, **wie** sie ausgeführt wird. Physische Gruppen oder gleiches Schema beweisen kein semantisches Sharing.

## Lösung

- Der Laufzeitplan aus #620 ordnet jedem Member Quellrepräsentations- und Permutations-ID zu; #619 projiziert logische Permutationen auf jedes vorhandene Schema.
- Der Executor partitioniert nach Device, Stype, Schema und Quellgruppe und stapelt je Partition eindeutige Indizes zu `[P,C]`.
- Für ein vollständiges Produkt wird `[S,R,C]` als View erweitert und per `gather` zu `[P,S,R,C]`. Beliebige, auch unabhängige K=E-Zuordnungen werden pro Quelle gebatcht; Quelltensoren werden nicht vorab kopiert.
- Unterschiedliche Spaltenzahlen bleiben getrennt. Fertige Gruppen/Locations werden intern direkt zusammengesetzt. Tensorzustände liegen in `BufferList`, IDs in `extra_state`; der per-Member-Pfad bleibt Fallback.

## Ausführbares Akzeptanzbeispiel

```python
import torch
from sdm import Stype, TableTensor, CategoricalTensor
from sdm.models import TabICLv2
from sdm.processing.execution import RecipeExecution

x = TableTensor.from_tensor(torch.randn(128, 32, device="cuda"))
y = TableTensor(categorical=CategoricalTensor(
    code=torch.arange(128, device="cuda", dtype=torch.int32).remainder(5)[:, None],
    categories=(torch.arange(5, device="cuda"),),
))
members = RecipeExecution(TabICLv2.default_recipe()).fit_transform(
    x, y, None, num_members=8,
    generator=torch.Generator(device="cuda").manual_seed(7),
)
orders = [member.x.columns[Stype.numerical] for member in members]
assert all(orders[i] == orders[i + 1] for i in range(0, 8, 2))
assert any(orders[i] != orders[i + 2] for i in range(0, 6, 2))
```

Auf `main` scheitert die Paarbedingung. Nach #619–#621 teilen nur geplante Paare eine Permutation; E bleibt dynamisch.

## GPU-Benchmark-Ergebnisse

NVIDIA L4 (23,66 GB), PyTorch `2.9.1+cu128`, CUDA 12.8, float32, E=8/S=2/C=100; 5 Warm-ups/30 synchronisierte Läufe. Fit und Input-Erzeugung sind im Transform-Test ausgeschlossen. K=4 ist der Worst-Case ohne zufällige Duplikate; Kandidaten sind Benchmark-Prototypen.

| 50.000 Zeilen                    |      Median/p95 |       Peak |
| -------------------------------- | --------------: | ---------: |
| Main, voller Processor, K=8      | 7,753/11,851 ms | 198,15 MiB |
| Geplant+gebatcht, K=4            |  1,632/1,721 ms | 152,59 MiB |
| Nur Ausführung, batched K=8      |  1,753/1,854 ms | 152,59 MiB |
| Nur Ausführung, cartesian K=4    |  1,636/1,689 ms | 152,59 MiB |
| Ein `gather` mit Quellkopie, K=8 |  3,062/3,189 ms | 305,18 MiB |

Der volle Kandidat ist **4,75×** schneller und spart 45,56 MiB. K=4 statt K=8 erklärt bei dieser speicherbandbreitenlimitierten Größe nur **6,7 %**; bei 1.000 Zeilen sind es launch-limitiert 48,8 %. Batching reduziert 34→5 CUDA-Events, 29→5 Launches und 5→0 Kopien. Ein einziges `gather` mit Quellauswahl ist 1,75× langsamer und benötigt 2× Speicher; pro Quelle zu batchen ist deshalb der robuste Algorithmus.

| Integration                                 |        Main Median/p95 |               Kandidat | Gewinn |
| ------------------------------------------- | ---------------------: | ---------------------: | -----: |
| Recipe 3.000×100, 10 kategorial, 10 Klassen |       60,473/86,219 ms |       41,881/51,157 ms |  1,44× |
| Recipe 50.000×100                           |     106,228/123,229 ms |       87,840/96,679 ms |  1,21× |
| Modell: 2.400 Kontext + 600 Query           | 2.417,990/2.478,307 ms | 2.395,339/2.420,758 ms | 1,009× |

Im Modell dominieren Transformer-Kosten; real bleibt knapp 1 % E2E. `ShuffleColumns` selbst skaliert bei 10.000×32 stark mit E (1,541 ms bei E=1; 13,126 ms bei E=16). Ausführungs-Batching ist daher allgemein sinnvoll, Sharing nur mit explizitem Modellplan.

Reproduktion: `python benchmark/ensemble_permutation_gpu.py --output benchmark/results/ensemble_permutation_gpu.json`. Die [JSON](benchmark/results/ensemble_permutation_gpu.json) enthält alle Samples, CUDA-Events, Peaks und Profilerzählungen.

## Testing

- Ergebnisse/Inverse gegen per-Member für K=1/2/4/8 und unabhängiges K=E vergleichen.
- Gleiche, unterschiedliche und überlappende Schemata; gerade/ungerade E; CPU und CUDA.
- Leeren Processor per `state_dict` laden; Outputs, IDs, Locations und Device exakt vergleichen.
- 1.000/10.000/50.000 Zeilen sowie Recipe-/Modell-E2E dauerhaft behalten.
