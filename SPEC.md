# Spec: Apply planned shuffles to stored table groups

Reference: `tabicl==2.0.0` at `f719c886a586ed4a29236345e319ac1ea596c478`. Baseline: `main` at `ee56f84161cb1c93b559977d67447e82b16d3541`.

## Problem

An `EnsembleTable` has logical estimators and stored tables. In the current TabICLv2 feature recipe, the `none`/`power` `Choice` produces eight logical estimators backed by one compatible group with exactly two stored tables: shape `[2, rows, columns]` and alternating locations. The reference selects four shuffle configurations and applies each to both normalizations.

Current `ShuffleColumns` instead fits and applies one state per logical estimator, producing eight temporary tables before `from_tables()` stacks normalization pairs again. Equal schema alone is not permission to share a permutation: sharing is correct only when the estimator plan assigns the same permutation ID. Different schemas or column counts must remain separate.

Simply copying the existing category grouping pattern is insufficient because `gather_members()` currently extracts grouped results and restacks them. That prototype regressed the representative CPU case.

## Proposed solution

- Deduplicate fitted column permutations by their exact index tuple and retain one permutation ID per logical estimator. Consume the explicit IDs produced by the Latin/category estimator-plan specs.
- For every permutation ID, select its logical members and apply the permutation once to each compatible stored group. In TabICLv2 this means four operations over two stacked normalization tables instead of eight operations over individual tables.
- Add a narrow internal `EnsembleTable` assembly path using the existing `groups` and `locations` vocabulary. It must preserve already-processed groups and logical member order without unpacking and restacking tensors; no new public container abstraction is needed.
- Use the grouped no-restack path initially on CUDA. Keep the current per-member CPU application because the large CPU case regressed despite fewer operations. Both paths use the same fitted permutation states and produce identical tables.
- Reuse the internal assembly path for paired target category shuffles only when it is non-regressive. Never combine incompatible groups, and validate permutation length once during fitting rather than in the hot path.

Pseudo-code:

```text
group logical members by planned permutation ID
for each ID: select referenced stored tables and permute each compatible group
record output group and (group, position) location for every logical member
return those groups and locations directly; do not restack
```

## Benchmark results

Synchronized wall time, 100 float32 columns, eight estimators, two stored tables, four shared permutations; GPU is an NVIDIA L4. Values are median/p95 over 10–20 runs.

|   Rows |            CPU current → grouped |     L4 current → grouped |
| -----: | -------------------------------: | -----------------------: |
|  1,000 |         6.39/6.91 → 2.50/2.55 ms | 6.61/7.05 → 1.17/1.48 ms |
| 10,000 |     16.30/25.06 → 14.19/20.84 ms | 6.68/6.98 → 1.16/1.46 ms |
| 50,000 | 120.71/142.14 → 151.73/164.83 ms | 7.39/7.93 → 1.68/1.70 ms |

At 50,000 rows, CUDA peak allocation drops from 312.6 to 152.6 MiB. The 25.7% CPU regression is why the initial fast path must be CUDA-only rather than selected solely from table compatibility.

## Testing

- Compare every member's data, columns, inverse transform, and order against the existing path for shared tables, two normalization tables, repeated mappings, and incompatible schemas.
- Verify four output groups of shape `[2, rows, columns]` for the representative plan without asserting private helper layout.
- Keep synchronized CPU/L4 scenarios at 1k, 10k, and 50k rows as permanent non-regression benchmarks.
