# Spec: Couple target category shuffles by estimator

Reference: `tabicl==2.0.0` at `f719c886a586ed4a29236345e319ac1ea596c478`. Baseline: `main` at `ee56f84161cb1c93b559977d67447e82b16d3541`.

## Problem

Current `ShuffleCategories` is already estimator-aware, but it samples each estimator independently. TabICLv2 does something different: it creates all cyclic class shifts, pairs them with the Latin feature patterns, shuffles those pairs with one seed, and executes every selected pair once with `none` and once with `power` normalization. The two normalization members must therefore share the same class shuffle.

This processor belongs only in the TabICLv2 target path. Categorical features are aligned, converted to numerical columns, and later covered by `ShuffleColumns`; moving `ShuffleCategories` into the feature path would multiply expensive categorical representations without matching the reference.

Independent shifts also leave output mapping ambiguous. Before estimator outputs are combined, every class axis must describe the same canonical, value-sorted target vocabulary.

## Proposed solution

- Build one private TabICLv2 estimator plan containing `(feature_pattern_id, class_pattern_id, normalization_id)` for every global estimator ID.
- Give `ShuffleCategories` a narrow internal resolver for fitted per-member permutations. It receives global member IDs and category counts; the existing `method="shift"` path remains unchanged when no plan is supplied.
- Generate class patterns exactly as the reference: identity for one estimator, otherwise every cyclic shift, then pair and seed-shuffle them with the feature pattern IDs before normalization is assigned.
- Reuse the processor's existing permutation-state deduplication so the same target mapping is transformed once and shared by the `none`/`power` pair.
- Align target categories by value and map each estimator output back to that canonical class axis before reduction. Do not add a public TabICLv2-specific mode to the generic processor.

Pseudo-code:

```text
fit canonical target vocabulary
build seeded feature × class configuration IDs
duplicate each selected pair for none and power
resolve the class permutation from the global estimator ID
transform the shared target once per unique class permutation
inverse-map every model output to canonical classes, then reduce
```

## Benchmark results

Synchronized wall time, 10 repeats, 50,000-row target, one categorical column, cardinality 100, eight estimators; the paired prototype has four unique shifts. GPU is an NVIDIA L4 with int32 codes.

| Device | Independent median/p95 | Paired median/p95 |      Peak delta |
| ------ | ---------------------: | ----------------: | --------------: |
| CPU    |        8.76 / 11.23 ms |    5.15 / 6.34 ms |    not measured |
| L4     |         6.97 / 7.22 ms |    3.85 / 4.01 ms | 2.54 → 2.15 MiB |

A deliberately irrelevant feature stress case (10,000 rows, 32 columns, cardinality 1,024) improved from 68.62 to 36.24 ms median on L4, but it must not be added to the TabICLv2 recipe.

## Testing

- Compare class-pattern assignment and the full coupled plan exactly with the pinned reference for classification and regression.
- Verify global IDs in vectorized and one-member sequential execution, shared `none`/`power` mappings, missing labels, one/many classes, and deterministic seeds.
- Compare per-estimator canonical outputs, reduction, final class columns, and public probabilities; keep generic independent-shuffle tests unchanged.
