# Ensemble-Aware Processing: Problem and Goals

## Context

A **Processor** $p$ is one processing step with configuration $c$ (its constructor parameters), optional fit-time randomness $\\omega$, and optional fitted state $\\theta$:

$$
\\theta = \\operatorname{fit}_p(X_{\\mathrm{fit}}; c, \\omega),
\\qquad
X' = \\operatorname{transform}\_p(X; \\theta, c, \\omega).
$$

$X\_{\\mathrm{fit}}$ contains context or training data only, never query data. `fit_transform(X)` is semantically `fit(X)` followed by `transform(X)`. An optional `inverse_transform` uses the same fitted state. Processors may be deterministic or stochastic, fitted or stateless, and may change the column schema.

A **Recipe** composes Processors for features, targets, related tables, and model outputs. For ensemble member $e$:

$$
R_e = p\_{k,e} \\circ \\dots \\circ p\_{1,e}.
$$

Members start from the same input but can diverge when a stochastic or member-specific step makes a different decision.

## Redundant member-isolated execution

Fitting a complete Recipe copy per member computes separate states and outputs:

$$
\\theta\_{i,e} = \\operatorname{fit}_i(X_{\\mathrm{fit},i,e}; c\_{i,e}, \\omega\_{i,e}),
\\qquad
X\_{i,e} = \\operatorname{transform}_i(X_{i-1,e}; \\theta\_{i,e}, c\_{i,e}, \\omega\_{i,e}).
$$

If fit input, transform input, Processor configuration, fit scope, and random decision are equivalent, the requests have the same state and output. Recomputing them per member wastes time and materializes copies too early. For RFM, repeating the feature Recipe for $E$ members and $T$ related tables can require up to $E(T+1)$ feature fits.

## Valid reuse and speed of light

For operation $o$ of Processor $i$, let $A\_{i,o}$ be all member requests and $G\_{i,o}=A\_{i,o}/{\\sim\_{i,o}}$ their semantic equivalence classes. Each class needs one computation; distinct compatible classes may share a leading tensor dimension.

The practical **speed of light** on a target device is the fastest valid plan:

$$
T\_{\\mathrm{SOL}} = \\min\_{P \\in \\mathcal{P}_{\\mathrm{valid}}} T_{\\mathrm{device}}(P)
\\approx \\sum_i \\sum_o \\sum\_{g \\in G\_{i,o}} C^\*\_{i,o}(g) + \\varepsilon.
$$

$C^\*\_{i,o}(g)$ is the fastest practical device execution for one equivalence class. $\\varepsilon$ contains unavoidable orchestration, kernel launches, transfers, and synchronization.

A plan is valid only if it preserves:

- fitted states and numerical outputs within an appropriate dtype-specific tolerance;
- context-only fit scope and separation from query data;
- externally observable RNG behavior and stable member order;
- column schema/order, stypes, dtypes, and device;
- target inverse/class mappings before estimator reduction.

Equivalence comes from execution provenance, never tensor-value comparison or content hashing. A processing step may preserve or split an equivalence class; only `ReduceEstimators` may aggregate members.

## TabICLv2 reference case

The feature path is:

```text
AlignCategories → ToNumerical
ImputeMean → DropConstantColumns → Standardize → Clip
→ Choice(Identity, PowerTransform) → ClipSigma → ShuffleColumns
```

For eight members with four `Identity` and four `PowerTransform` choices, the shared deterministic prefix runs once, `PowerTransform` runs once for its equivalent branch, and member-specific permutations are created only at `ShuffleColumns`. Query data reuses the fitted branches and mappings. Classification also preserves class permutations; regression applies fitted target inverse transformation before reduction.

With shared work $D$, power work $P$, branch work $B_0,B_1$, and required member-specific work $R$:

$$
T\_{\\mathrm{isolated}} \\approx 8D + 4P + 4B_0 + 4B_1 + 8R,
$$

$$
T\_{\\mathrm{shared}} \\approx D + P + B_0 + B_1 + R_E^\* + \\varepsilon.
$$

$R_E^\*$ is vectorized execution of genuinely distinct member work, not reuse of unequal results.

## RFM reference case

Task features, target, and each related table are separate fit scopes. State may be shared among equivalent members within one table, but not across logical tables. A `Choice` is therefore sampled and stored independently for each concrete table scope. Relationships and task links remain shared metadata.

## Scope and acceptance

- Version 1 supports row-preserving Processors, fit-time randomness, immutable post-fit state, one device per ensemble, and different context/query row counts. A Recipe with variable-schema processors accepts one unbatched logical table per call; batching logical tables with independently changing schemas is out of scope.
- Different output schemas are supported by separate physical groups; row-changing Processors and multi-GPU execution are out of scope.
- Every supported leaf either processes leading variants independently or implements the structural ensemble contract; unsupported external Processors fail clearly.
- TabICLv2 must match its pinned eight-estimator reference for preprocessing, public forward, and cached fit/predict in classification and regression.
- RFM must preserve parallel/sequential and direct/cached behavior on a canonical relational dataset containing IDs, keys, datetimes, numerical/categorical values, missing values, constants, outliers, and unseen query categories.
- The target processing workload is 40k context rows, 10k query rows, 100 features, float32, and eight members split into four power and four identity variants.
- Timings exclude dataset construction and correctness checks. Kernel timing excludes transfers; host/device transfers and synchronization are measured separately.
- Report median, p95, throughput, hardware, dtype, and peak memory. The fastest reproducible valid GPU plan is the empirical lower bound; no fixed runtime is assumed.
