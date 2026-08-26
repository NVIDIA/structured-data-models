# Spec: Estimatorgekoppelte Target-Kategorie-Shuffles

Referenz: `tabicl==2.0.0` bei `f719c886a586ed4a29236345e319ac1ea596c478`. Baseline: `main` bei `bebaef1ccd074a376dd2f4c3e5c0c010fb8d6db3`.

## Problem

`ShuffleCategories` arbeitet auf `main` bereits estimatorabhängig, zieht aber für jeden logischen Estimator unabhängig eine Permutation. TabICLv2 bildet zuerst Paare aus Feature- und Klassen-Pattern, mischt diese mit einem Seed und führt jedes ausgewählte Paar einmal mit `none`- und einmal mit `power`-Normalisierung aus. Beide Normalisierungen müssen deshalb denselben Klassen-Shift verwenden.

Der Processor gehört im TabICLv2-Rezept nur in den Target-Pfad. Kategoriale Features werden ausgerichtet, numerisch kodiert und später durch `ShuffleColumns` abgedeckt. Ein zusätzlicher Feature-Category-Shuffle würde Repräsentationen vervielfachen, ohne Referenzverhalten herzustellen. Vor der Estimatorreduktion muss außerdem jede Modell-Ausgabe wieder dieselbe kanonische, nach Werten sortierte Klassenachse beschreiben.

## Lösung

- Die Position eines logischen Members relativ zu seiner gespeicherten Tabelle definiert die Wiederholung. Tabellen mit demselben kategorialen Schema erhalten für dieselbe Wiederholung denselben Zustand; unterschiedliche Kardinalitäten bleiben getrennt. Damit bleibt die Regel allgemein und benötigt keinen öffentlichen TabICLv2-Modus.
- Für `method="shift"` enthält ein Zustand nur Offsets, inverse Kategorieordnungen, Blockgrenzen und Divisoren statt eines vollständigen Mappings pro Member. `random` behält sein allgemeines Mapping.
- Alle dynamischen Zustandstensoren liegen als rekursive `BufferList` vor. Die Zuordnung der logischen Member zu Zuständen wird über `get_extra_state()`/`set_extra_state()` gespeichert; Fitted-Status und Device-Transfer folgen dem aktuellen `Processor`-Vertrag. Ein leer konstruierter Processor kann den Zustand daher mit `load_state_dict()` ohne erneutes Fitten rekonstruieren.
- Kompatible Target-Tabellen werden für alle eindeutigen Offsets auf CUDA in einem Durchlauf berechnet: `(code[None] - shifts[:, None, None]) % K`. Ein `where` erhält negative Missing-Codes; die invers verschobenen Kategorie-Vektoren sichern dieselben dekodierten Werte. `random` nutzt weiterhin das allgemeine Mapping.
- Nach dem Modell wird jede Estimatorausgabe mit ihrem inversen Shift auf den kanonischen Klassenraum abgebildet und erst dann reduziert.

## Ausführbares Akzeptanzbeispiel

Dieses Beispiel wurde auf CUDA gegen `main` und den implementierten Stand dieses PRs ausgeführt.

```python
import torch
import sdm
from sdm.processing import ShuffleCategories
from sdm.tensor import EnsembleTable

device = torch.device("cuda")
def table(start):
    return sdm.TableTensor(categorical=sdm.CategoricalTensor(
        code=torch.arange(7, dtype=torch.int32, device=device).unsqueeze(1),
        categories=(torch.arange(start, start + 7, device=device),),
    ))

ensemble = EnsembleTable.from_tables(
    tables=(table(0), table(10)), member_table_ids=(0, 1, 0, 1)
)
processor = ShuffleCategories(method="random").fit_ensemble(
    ensemble, generator=torch.Generator(device=device).manual_seed(7)
)
output = processor.transform_ensemble(ensemble)
restored = ShuffleCategories(method="random")
restored.load_state_dict(processor.state_dict())
again = restored.transform_ensemble(ensemble)
pairs = ((0, 1), (2, 3))
paired = tuple(torch.equal(output.table(a).categorical.code, output.table(b).categorical.code) for a, b in pairs)
state_ok = all(output.table(i).equal(again.table(i)) for i in range(4))
print(f"paired={paired}, state_dict={state_ok}")
```

Ausgabe auf `main`: `paired=(False, False), state_dict=True`. Ausgabe nach diesem PR: `paired=(True, True), state_dict=True`.

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
- Nach Fit den `state_dict` in einen leeren Processor laden und Fitted-Status sowie alle Member-Ausgaben für `shift` und `random` exakt vergleichen.
- Per-Estimator-Ausgaben nach kanonischer Rückabbildung, Reduktion, finale Klassenreihenfolge und öffentliche Wahrscheinlichkeiten vergleichen.
