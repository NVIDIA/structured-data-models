# Ensemble-Aware Processing: Problem and Goals

## Context

A **Processor** $p$ is a processing step with configuration $c$, an optional
fit-time random variant $\\omega$, and optional fitted state $\\theta$:

$$
\\theta = \\operatorname{fit}_p(X_{\\mathrm{fit}}; c, \\omega),
\\qquad
X' = \\operatorname{transform}\_p(X; \\theta, c, \\omega).
$$

$X\_{\\mathrm{fit}}$ contains context or training data only, never query data.
`fit_transform(X)` is semantically equivalent to `fit(X)` followed by
`transform(X)`. An optional `inverse_transform` uses the same fitted state.
Processors can be stateless or fitted, deterministic or stochastic, and may
change the number of columns or the column schema.

A **Recipe** is an ordered composition of Processors. For ensemble member $e$:

$$
R_e = p\_{k,e} \\circ \\dots \\circ p\_{1,e}.
$$

All members may start with the same input but take different execution paths
because of different configurations or fit-time random variants.

## Problem

SDM currently copies and fits the complete Recipe for every member. Processor
step $i$ and member $e$ therefore produce separate states and outputs:

$$
\\theta\_{i,e} =
\\operatorname{fit}_i(X_{\\mathrm{fit},i,e}; c\_{i,e}, \\omega\_{i,e}),
\\qquad
X\_{i,e} =
\\operatorname{transform}_i(
X_{i-1,e}; \\theta\_{i,e}, c\_{i,e}, \\omega\_{i,e}
).
$$

When the fit input, transform input, configuration, fit scope, and random
variant are semantically equal, the state and output are equal as well. The
current execution still computes these equivalent requests repeatedly:

$$
T\_{\\mathrm{current}} =
\\sum\_{e=1}^{E}\\sum\_{i=1}^{k}
\\left(C\_{\\mathrm{fit}}(i,e) + C\_{\\mathrm{transform}}(i,e)\\right).
$$

For RFM, the feature pipeline is additionally copied and fitted for every
related table. With $E$ members and $T$ related tables this can require up to
$E,(T+1)$ feature fits, even though deterministic work is often identical
across members within the same table.

## Goal and Speed of Light

The **speed of light** is the minimum runtime of a valid execution plan $P$ on
the target hardware:

$$
T\_{\\mathrm{speed\\ of\\ light}} =
\\min\_{P \\in \\mathcal{P}_{\\mathrm{valid}}} T_{\\mathrm{target\\ hardware}}(P).
$$

For Processor step $i$ and operation $o$, such as `fit`, `transform`,
`fit_transform`, or `inverse_transform`, let $A\_{i,o}$ be the set of all
requests and

$$
G\_{i,o} = A\_{i,o} / {\\sim\_{i,o}}
$$

the set of their semantic equivalence classes. Each class $g \\in G\_{i,o}$ only
needs to be computed once. Distinct but compatible classes may be vectorized:

$$
T\_{\\mathrm{speed\\ of\\ light}} \\approx
\\sum_i\\sum_o\\sum\_{g \\in G\_{i,o}} C^\*\_{i,o}(g) + \\varepsilon.
$$

$C^\*\_{i,o}(g)$ is the fastest practical GPU execution of the class.
$\\varepsilon$ covers unavoidable orchestration, transfers, kernel launches,
and synchronization. Fusion across Processor boundaries is allowed when the
resulting plan remains valid.

A plan is valid when it preserves:

- fitted states and numerical outputs within a dtype-specific tolerance;
- fit scope and the separation of context from query data;
- RNG order and externally observable stochastic behavior;
- column schema and order, stypes, dtypes, and device;
- member-specific mappings and output order.

Equivalence is derived from execution provenance, not from tensor comparison or
content hashing. During fit and transform, equivalence classes may only be
preserved or split. `EnsembleReduce` is the explicit aggregation boundary.

## TabICLv2 Reference Case

The current feature Recipe converts categorical features with
`CategoricalAlign → ToNumerical`, then processes all numerical features with:

```text
MeanImpute
→ ConstantFilter
→ StandardScale
→ Clip
→ Choice(Identity, Power)
→ SigmaClip
→ FeaturePermute
```

`MeanImpute`, `ConstantFilter`, and `StandardScale` learn deterministic state.
`ConstantFilter` may change the number of columns. `Choice` selects `Identity`
or `Power` during fit for every member. `SigmaClip` then receives
branch-specific inputs, while `FeaturePermute` creates member-specific
permutations.

For classification, the target Recipe uses
`CategoricalAlign → CategoryShuffle`; for regression it uses `StandardScale`.
The output Recipe applies member-specific inverse mappings and may aggregate
with `EnsembleReduce(method="mean")`.

For eight members with four `Identity` and four `Power` variants, the desired
execution is:

1. Compute the shared deterministic prefix once.
2. Preserve the same eight `Choice` decisions, grouped into two branches.
3. Fit and apply `Power` once for the equivalent `Power` class.
4. Fit and apply `SigmaClip` once per branch.
5. Preserve member-specific `FeaturePermute` variants and vectorize them where
   possible.
6. Transform query data with exactly the same fitted states and variants.

With shared prefix $D$, `Power` work $P$, branch-specific work $B_0$ and $B_1$,
and necessary member-specific work $R$:

$$
T\_{\\mathrm{current}} \\approx
8D + 4P + 4B_0 + 4B_1 + 8R,
$$

$$
T\_{\\mathrm{optimal}} \\approx
D + P + B_0 + B_1 + R_E^\* + \\varepsilon.
$$

$R_E^\*$ is the fastest vectorized execution of necessary distinct member work,
not reuse of identical results.

## RFM Reference Case

Task features, target, and every related table are separate fit scopes.
Deterministic prefixes and text or categorical encoders should be shared across
compatible members within the same table. Different tables do not share fitted
state in version 1. Context and query data for a table use the same fitted plan,
while `relationships` and `task_links` remain shared graph metadata.

Native RFM fits table-local statistics and creates table-local feature
permutations while consuming randomness from an estimator-owned generator. It
does not have a direct equivalent of TabICLv2's feature-level
`Choice(Identity, Power)`. Whether such a `Choice` in an RFM Recipe should be
owned by the estimator or by each table fit scope is therefore a design
decision, not existing RFM behavior.

Resetting the same seed for every table is not a sufficient semantic contract:
it correlates all other stochastic steps, makes results depend on Processor
order, and can still diverge when a table consumes a different number of random
draws. The implementation needs an explicit random plan or generator ownership
rule instead.

The Recipe still receives the complete graph so that future scheduling of
compatible operations across tables does not require an API change.

## Scope and Acceptance Criteria

- Version 1 supports row-preserving Processors, fit-time randomness, and fitted
  state that does not change during `transform`.
- Context and query may have different row counts. Query transformation must
  produce the fitted context schema and column order.
- Different schemas or column counts are allowed but execute separately.
- One ensemble execution stays on one device. Multi-GPU execution and
  row-count-changing Processors are out of scope.
- Fitted states and intermediate results are not shared across logical tables
  in version 1.
- Every supported row-preserving Processor either processes leading dimensions
  independently or implements its own ensemble contract for structural or
  stochastic behavior. There is no implicit per-member fallback.
- External Processors that satisfy neither contract are rejected.
- Strict TabICLv2 parity must be demonstrated for forward and `fit`/`predict`.
- Target workload: 40k context rows, 10k query rows, 100 features, and eight
  members split into four `Power` and four `Identity` variants.
- The measured path starts at Recipe processing and ends after model execution.
  It includes processing, required member-ordered model-input materialization,
  model execution, and output transformation. Dataset creation and external
  host/device transfers are excluded.
- The previously reported 0.46–0.47 seconds and target of 0.19–0.20 seconds
  must be measured again using this common boundary.
- Report median, p95, kernel time, transfers, synchronization, peak memory, and
  throughput. Dataset construction and correctness checks are excluded from
  timed regions.

The practical speed of light is measured empirically on the target GPU using
preallocated GPU-resident inputs, warm-up, CUDA events, and explicit
synchronization. Compare valid execution plans, layouts, vectorization,
fusion, and dtypes under the same numerical tolerance. The fastest reproducible
variant, including median and p95, is the measured lower bound. GPU model,
software versions, shapes, dtype, and data characteristics are part of the
result.

## Open Questions

- **P1 — RFM stochastic ownership:** Should feature-level `Choice` be selected
  once per estimator and reused across tables, or selected independently for
  each table fit scope?
- **P2 — Measured speed of light:** What end-to-end runtime does the best valid
  GPU baseline achieve on the available target GPU?
