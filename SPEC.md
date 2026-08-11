# Spec: Lazy Latin column permutations for TabICLv2

Reference: `tabicl==2.0.0` at `f719c886a586ed4a29236345e319ac1ea596c478`. Baseline: `main` at `ee56f84161cb1c93b559977d67447e82b16d3541`.

## Problem

TabICLv2 uses `Shuffler(method="latin")` for feature columns up to 4,000 columns and random permutations above that limit. Current `ShuffleColumns` supports only `shift` and `random`, and the TabICLv2 recipe selects `shift`, so it cannot reproduce the reference ensemble. There is no probability distribution to copy: the reference builds a deterministic randomized Latin square with Python `random.Random(seed)`.

A literal port is unnecessarily expensive because it constructs all `C × C` Python indices before the eight estimators are selected. At `C=4,000`, that takes 6.41 s and reaches 397.7 MiB RSS in the measured standalone process.

## Proposed solution

- Add `method="latin"` to `ShuffleColumns`; preserve the existing `shift` and `random` paths unchanged.
- Reproduce the reference RNG calls, but store the Latin square lazily as three one-dimensional sequences: the chosen symbol order, shuffled row IDs, and shuffled column IDs.
- Let the estimator plan combine Latin pattern IDs with class-shuffle IDs and normalization IDs, then materialize only the selected column permutations as device-local `torch.long` tensors.
- Keep the reference behavior for one estimator and the `C > 4,000` random fallback.
- Do not expose a probability parameter. The only new public name is the established `method="latin"`; plan plumbing remains internal until another model needs it.

Pseudo-code:

```text
seed Python RNG
draw symbol order; shuffle row IDs; shuffle column IDs
shuffle lightweight (feature ID, class ID) pairs
select requested estimators and normalizations
materialize only their feature permutations on the input device
```

## Benchmark results

Planning was measured on CPU; application is covered by the grouped-application spec. The lazy prototype reproduced the complete reference feature × class × normalization plan exactly for multiple sizes, including `C=100`, 10 classes, 8 estimators, seed 42.

| Columns |             Eager reference |   Lazy selected-8 prototype |                         Peak RSS |
| ------: | --------------------------: | --------------------------: | -------------------------------: |
|     100 | 0.473 / 0.636 ms median/p95 |            0.140 / 0.169 ms |                       negligible |
|   4,000 |         6,407 ms single run | 38.04 / 49.45 ms median/p95 | 397.7 MiB → 17.2 MiB process RSS |

At 4,001 columns, the required random fallback generated eight permutations in 7.36 ms median.

## Testing

- Compare permutations exactly with the pinned reference across seeds, column counts, one/many estimators, and the 4,000/4,001 boundary.
- Assert the Latin property and exact full-plan selection, not private helper structure.
- Keep existing `shift`/`random`, inverse-transform, CPU, and CUDA tests unchanged.
