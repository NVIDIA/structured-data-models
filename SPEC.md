# Spec: Skalierbare Latin-Spaltenpermutationen

Referenz: `tabicl==2.0.0` bei `f719c886a586ed4a29236345e319ac1ea596c478`. Baseline: `main` bei `2a73246320ea4188d8b8791e136ac8f111e112d6`.

## Problem

`ShuffleColumns` unterstützt `random` und `latin`. TabICLv2 verwendet Latin-Permutationen, fiel zuvor aber oberhalb von 4.000 Spalten auf `random` zurück. Diese Grenze schützt nur die rekursive `O(C²)`-Referenzimplementierung; sie ist keine allgemeine Qualitäts- oder Bibliotheksregel. Ein explizites `method="latin"` darf deshalb nicht still seine Semantik ändern.

Gruppen einer `EnsembleTable` können unterschiedlich viele, aber stark überlappende Spalten haben. Sie sollen denselben **logischen** Zufallsplan verwenden, ohne einen falsch dimensionierten Permutationstensor zu teilen.

## Lösung

- `random` bleibt Default; `latin` ist explizit. Ressourcenlimits melden einen Fehler, wechseln aber nie automatisch die Methode.
- Der Modell-/Recipe-Plan wird erst erzeugt, wenn Estimatoranzahl und Schemata bekannt sind. Vereinfacht:
  1. Erzeuge eine gemeinsame zufällige Reihenfolge aller bekannten Spalten und Latin-Pattern-IDs.
  2. Entferne für jede Gruppe die Spalten, die dort nicht existieren.
  3. Nummeriere die übrigen Positionen lokal von `0…C-1`.
  4. Materialisiere nur die benötigten Zeilen: `perm[p,j] = symbols[(pattern[p] - column_order[j]) % C]`.
- Damit bleiben gemeinsame Spalten logisch gekoppelt; konkrete `[P,C]`-Tensoren bleiben schemaspezifisch.
- Für exakte TabICLv2-Seeds dekodiert ein interner Order-Statistic-Plan die Python-RNG-Ziehungen in `O(C log C)`. Der generische CUDA-Pfad kennt weder TabICL noch die 4.000er-Grenze.
- Tensorzustand liegt in `BufferList`, logische IDs in `extra_state`; `state_dict` stellt einen gefitteten Processor ohne erneutes Fitten wieder her.

## Ausführbares Akzeptanzbeispiel

```python
import torch
import sdm
from sdm.processing import ShuffleColumns
from sdm.tensor import EnsembleTable

columns = ("a", "b", "c", "d")
x = sdm.TableTensor.from_tensor(
    torch.arange(8, dtype=torch.float32, device="cuda").reshape(2, 4), columns=columns
)
p = ShuffleColumns(method="latin").fit_ensemble(
    EnsembleTable(x, num_members=4),
    generator=torch.Generator(device="cuda").manual_seed(7),
)
out = p.transform_ensemble(EnsembleTable(x, num_members=4))
orders = [out.table(i).columns[sdm.Stype.numerical] for i in range(4)]
assert all({row[i] for row in orders} == set(columns) for i in range(4))
restored = ShuffleColumns(method="latin")
restored.load_state_dict(p.state_dict())
again = restored.transform_ensemble(EnsembleTable(x, num_members=4))
assert all(out.table(i).equal(again.table(i)) for i in range(4))
```

Auf `main` scheitert bereits `method="latin"`; nach der Implementierung bestehen Latin-Invariante und `state_dict`-Roundtrip.

## GPU-Benchmark-Ergebnisse

NVIDIA L4 (23,66 GB), PyTorch `2.9.1+cu128`, CUDA 12.8; 5 Warm-ups, 30 Wiederholungen (Plan-Gesamtzeit: 10), CUDA-synchronisierte Wall-Time, Peak-Allokation über dem Ausgangswert. Der kompakte exakte Plan stimmt in **400/400 Fällen** (`seed=0…3`, `C=1…100`) elementweise mit der gepinnten Referenz überein.

| Spalten | Exakter Plan + CUDA, K=8 Median/p95 | CUDA-Materialisierung | State K=8 |
| ------: | ----------------------------------: | --------------------: | --------: |
|     100 |                      0,621/0,655 ms |              0,380 ms |  0,02 MiB |
|   4.000 |                    14,194/16,401 ms |              1,325 ms |  0,55 MiB |
|  16.000 |                    62,488/69,416 ms |              4,163 ms |  2,20 MiB |
|  64.000 |                  286,924/329,393 ms |             16,790 ms |  8,79 MiB |

Es gibt keinen Laufzeitsprung bei 4.000. Der serielle, exakt reproduzierte RNG-Plan dominiert; die CUDA-Materialisierung skaliert annähernd linear. Beim allgemeinen zyklischen CUDA-Plan kosten K=4 und K=8 bei 64.000 Spalten beide ≈0,20 ms, aber der Zustand verdoppelt sich von 3,91 auf 7,81 MiB. Weniger Permutationen sparen hier Speicher, nicht messbar Rechenzeit.

Reproduktion und Rohdaten: [`benchmark/ensemble_permutation_gpu.py`](https://github.com/NVIDIA/structured-data-models/blob/agent/spec-tabiclv2-grouped-shuffles/benchmark/ensemble_permutation_gpu.py) und [JSON](https://github.com/NVIDIA/structured-data-models/blob/agent/spec-tabiclv2-grouped-shuffles/benchmark/results/ensemble_permutation_gpu.json), Aufruf: `python benchmark/ensemble_permutation_gpu.py --output benchmark/results/ensemble_permutation_gpu.json`.

## Testing

- Latin-Invarianten, Seeds, kleine/große `C`, unterschiedliche und überlappende Schemata.
- Exakte Referenzparität des internen TabICLv2-Plans; kein Methodenwechsel oberhalb 4.000.
- Transformation, Inverse, Device-Wechsel und leerer `state_dict`-Roundtrip.
