# Ensemble-aware processing design

This document proposes an interface for reusing fitted preprocessing state across
ensemble members when doing so is equivalent to the current per-member execution.
It is a design proposal only; it does not change runtime behavior.

## Problem

The shared model path currently clones and fits the full recipe once per
estimator. In `ICLModel.forward`, the model resolves the default or user recipe,
deep-copies it `num_estimators` times, and then calls
`recipe.features.fit_transform(...)`, `recipe.target.fit_transform(...)`, and
`recipe.features.transform(...)` inside the estimator loop
(`sdm/models/base.py:111-124`). Related tables add another table-local set of
deep-copied feature processors inside that same estimator loop
(`sdm/models/base.py:126-149`). The cached `fit`/`predict` path has the same
shape: `fit` deep-copies one recipe per estimator and fits each copy
(`sdm/models/base.py:222-247`), while `predict` loops over the cached member
recipes and transforms query data per member (`sdm/models/base.py:329-346`).

That behavior is correct, but it repeats deterministic preprocessing before any
estimator-specific randomness exists. The base processor API is intentionally
simple: `fit` learns state, `transform` applies it, and `fit_transform` is
defined as `fit(...).transform(...)` (`sdm/processing/base.py:75-128`).
`Sequential` follows that same contract by running each step's
`fit_transform(...)` in order (`sdm/processing/common/sequential.py:36-76`).
The missing abstraction is an execution context that can say "these ensemble
members are still equivalent; fit this prefix once" and then expand only at the
step that introduces estimator-specific state.

The TabICLv2 default recipe shows why this matters. Its numerical feature route
runs a deterministic prefix (`MeanImpute`, `ConstantFilter`, `StandardScale`,
`Clip`) before `Choice(Identity(), Power())`, then `SigmaClip`, then
`FeaturePermute` (`sdm/models/tabiclv2/recipe.py:41-49`). Today every cloned
recipe pays that prefix independently. If several estimators choose `Power`, they
also independently fit equivalent `Power` states on equivalent input.

The observed target workload is 40k context rows, 10k query rows, 100 features,
and 8 estimators split as 4 power-normalized members plus 4 no-normalization
members. Current SDM GPU recipe/preprocessing overhead is about 0.46-0.47 s. The
shared-state target derived from the stage timings is about 0.19-0.20 s with
strict parity.

## Reference behavior from original TabICL

Original TabICL already groups by normalization method instead of fitting the
preprocessing pipeline once per estimator. In the upstream implementation at
`soda-inria/tabicl@46b91961`, `EnsembleGenerator.fit` defaults
`norm_methods` to `["none", "power"]`, generates grouped ensemble configs, and
then fits one `PreprocessingPipeline` per normalization key:

- default normalization methods:
  [`src/tabicl/_sklearn/preprocessing.py#L1037-L1043`](https://github.com/soda-inria/tabicl/blob/46b91961db4f8873dd049ec09990698a435e1e29/src/tabicl/_sklearn/preprocessing.py#L1037-L1043)
- one fitted preprocessor per grouped `norm_method`:
  [`src/tabicl/_sklearn/preprocessing.py#L1063-L1073`](https://github.com/soda-inria/tabicl/blob/46b91961db4f8873dd049ec09990698a435e1e29/src/tabicl/_sklearn/preprocessing.py#L1063-L1073)
- grouping configurations by normalization method:
  [`src/tabicl/_sklearn/preprocessing.py#L1112-L1128`](https://github.com/soda-inria/tabicl/blob/46b91961db4f8873dd049ec09990698a435e1e29/src/tabicl/_sklearn/preprocessing.py#L1112-L1128)

The fitted `PreprocessingPipeline` is where `power` maps to sklearn's
Yeo-Johnson `PowerTransformer` and where the normalization fit is paid:

- construct `PowerTransformer(method="yeo-johnson", standardize=True)`:
  [`src/tabicl/_sklearn/preprocessing.py#L706-L710`](https://github.com/soda-inria/tabicl/blob/46b91961db4f8873dd049ec09990698a435e1e29/src/tabicl/_sklearn/preprocessing.py#L706-L710)
- call `normalizer_.fit_transform(...)` once for the fitted pipeline:
  [`src/tabicl/_sklearn/preprocessing.py#L727-L729`](https://github.com/soda-inria/tabicl/blob/46b91961db4f8873dd049ec09990698a435e1e29/src/tabicl/_sklearn/preprocessing.py#L727-L729)

During transform, TabICL reuses the grouped fitted preprocessor and applies the
member-specific feature and class shuffles afterwards:

- reuse `self.preprocessors_[norm_method].X_transformed_` for training mode:
  [`src/tabicl/_sklearn/preprocessing.py#L1190-L1208`](https://github.com/soda-inria/tabicl/blob/46b91961db4f8873dd049ec09990698a435e1e29/src/tabicl/_sklearn/preprocessing.py#L1190-L1208)
- transform query data once per normalization method:
  [`src/tabicl/_sklearn/preprocessing.py#L1219-L1231`](https://github.com/soda-inria/tabicl/blob/46b91961db4f8873dd049ec09990698a435e1e29/src/tabicl/_sklearn/preprocessing.py#L1219-L1231)
- for combined train/query mode, concatenate the grouped preprocessed train and
  test arrays before applying each shuffle config:
  [`src/tabicl/_sklearn/preprocessing.py#L1233-L1245`](https://github.com/soda-inria/tabicl/blob/46b91961db4f8873dd049ec09990698a435e1e29/src/tabicl/_sklearn/preprocessing.py#L1233-L1245)

The SDM design should preserve the general idea while avoiding a TabICL-specific
or `Power`-specific cache.

## Proposed interface

Keep the existing public `Processor.fit`, `Processor.transform`, and
`Processor.fit_transform` APIs unchanged. Add an internal ensemble-aware
execution API that processors can opt into. Processors that do not opt in use the
current per-estimator fallback.

```python
from dataclasses import dataclass
from typing import Hashable, Literal

@dataclass(frozen=True)
class EnsembleContext:
    num_members: int
    role: Literal["features", "target", "related_features", "output"]
    fit_scope: Literal["context", "target", "related_context"]
    table_key: Hashable | None
    generator: torch.Generator | None

@dataclass(frozen=True)
class FitStateKey:
    processor_path: str
    processor_fingerprint: Hashable
    fit_scope: str
    table_key: Hashable | None
    schema_fingerprint: Hashable
    device: torch.device
    dtype: torch.dtype | None
    rng_key: Hashable | None

@dataclass
class EnsembleTable:
    table: TableTensor
    # Maps logical estimator index [0, E) to a row in the leading member/group
    # dimension of `table`. A single shared table has one row and all members
    # map to 0.
    member_to_group: torch.Tensor
```

Optional processor hooks:

```python
class Processor:
    ensemble_shareable_fit: ClassVar[bool] = False
    ensemble_mutates_after_fit: ClassVar[bool] = False

    def fit_state_key(
        self,
        table: TableTensor,
        *,
        context: EnsembleContext,
        processor_path: str,
    ) -> FitStateKey | None:
        return None

    def fit_transform_ensemble(
        self,
        table: EnsembleTable,
        *,
        context: EnsembleContext,
        processor_path: str,
    ) -> EnsembleTable:
        return _fallback_per_member_fit_transform(self, table, context)

    def transform_ensemble(
        self,
        table: EnsembleTable,
        *,
        context: EnsembleContext,
        processor_path: str,
    ) -> EnsembleTable:
        return _fallback_per_member_transform(self, table, context)
```

The main executor should live outside model families, for example as an internal
helper used by `ICLModel.forward` and `ICLModel.fit`. `num_estimators` should
remain on `Model.forward`/`Model.fit`, because it controls model ensembling, not
the recipe definition. The model constructs an `EnsembleContext` and passes it
through the processing executor. `Recipe` should not permanently store
`num_estimators`.

## Execution semantics

1. Start with one logical shared group. For a normal feature table this is
   represented as `EnsembleTable(table=x_context, member_to_group=zeros(E))`.
   The physical tensor does not need to materialize a leading dimension until a
   processor needs it.
2. Deterministic processors with a valid `FitStateKey` are fitted once per
   equivalent key. Their transformed output stays shared while all members still
   map to the same group.
3. `Sequential` becomes the first container that understands `EnsembleTable`.
   It passes the current grouped table through each step. It does not need to
   know model-family details.
4. `Choice` samples the selected option for every logical member using the same
   RNG semantics as today's repeated fits. It then groups members by selected
   option, computes each unique selected branch once for each equivalent input
   group, and returns a table whose `member_to_group` reconstructs estimator
   order. The current `Choice` draws one scalar choice in `_fit` and fits only
   that branch (`sdm/processing/common/choice.py:47-62`); the ensemble hook is
   the grouped generalization of that behavior.
5. `FeaturePermute` and `CategoryShuffle` are member-specific transformation
   steps. They should be able to create an estimator/member dimension and apply
   all permutations in one grouped tensor operation when practical.
   `FeaturePermute` currently stores one permutation in `_fit` and applies it in
   `_transform` (`sdm/processing/common/feature_permute.py:38-67`).
   `CategoryShuffle` similarly fits per-column categorical permutations
   (`sdm/processing/categorical/category_shuffle.py:47-90`).
6. Query transforms reuse the fitted grouped states from context fitting. They
   may still expand at stochastic transforms whose fitted state is
   member-specific.
7. Output processors already receive stacked estimator outputs and may reduce
   the leading estimator dimension. The existing recipe docs state that member
   outputs enter `recipe.output` as `[E, ..., R, O]`
   (`sdm/processing/recipe.py:96-101`).

## Reuse rules

Reuse is allowed only when the current implementation would have supplied the
same fit input and processor configuration.

A reusable fitted state key must include:

- processor path inside the recipe, not only processor type;
- processor configuration/fingerprint;
- fit scope (`features`, `target`, `related_features`) and leakage boundary;
- table key for related-table processing;
- schema fingerprint, including semantic type and column order;
- input device and dtype for tensor-owned state;
- RNG choice or branch identity for stochastic processors;
- a marker that the processor does not mutate fitted state after fitting.

Processors are not shareable by default. That default keeps correctness simple:
unknown processors fall back to today's deep-copy/per-estimator execution.

## TabICL example

For 8 estimators split as 4 power-normalized members plus 4 no-normalization
members:

1. Run the deterministic feature prefix once on context data:
   `MeanImpute -> ConstantFilter -> StandardScale -> Clip`.
2. At `Choice(Identity(), Power())`, sample choices for all 8 estimator members.
3. Compute the `Identity` branch once for the 4 no-normalization members.
4. Fit `Power` once for the 4 power-normalized members and transform that group.
5. Continue through `SigmaClip` once per resulting normalization group when its
   fit input is group-identical.
6. Expand to 8 members at `FeaturePermute`, because feature order is
   estimator-specific.
7. For classification targets, expand at `CategoryShuffle`, because class-code
   permutations are estimator-specific.

This matches the useful property of original TabICL: power normalization is paid
once for the power group, not once per power estimator. It also keeps the SDM
processing abstraction generic enough for `Quantile`, categorical processors,
and future expensive encoders.

## KumoRFM and related-table example

KumoRFM inherits the TabICLv2 default recipe and adds datetime feature encoding
to the first feature dispatch route (`sdm/models/kumorfm/recipe.py:13-25`).
The shared `ICLModel` path currently deep-copies feature processors per related
table inside each estimator loop (`sdm/models/base.py:126-149` and
`sdm/models/base.py:231-247`). After preprocessing, KumoRFM builds task graphs
from related context/query tables (`sdm/models/kumorfm/model.py:329-368`) and
then embeds each table by name (`sdm/models/kumorfm/model.py:373-417`).
`RelatedTables` stores tables keyed by table name and exposes a schema with the
same keys and relationship/task-link metadata (`sdm/relational/task.py:181-246`).

The interface should therefore reuse deterministic fitted prefixes per
equivalent table/input scope, not globally:

- fit a deterministic prefix once per `(table_key, fit_scope, schema, processor_path, config, device, dtype)`;
- reuse it across estimators for the same table when the input is equivalent;
- do not reuse it across different tables unless data/schema/config/scope are
  proven equivalent;
- for future text encoders, route support already exists through
  `StypeDispatch(text=...)` (`sdm/processing/common/stype_dispatch.py:43-44` and
  `sdm/processing/common/stype_dispatch.py:65-83`), so an expensive
  deterministic text encoder can opt into `fit_state_key` and be shared across
  estimators for the same table;
- keep relational graph construction table-scoped; preprocessing reuse should
  not mix relationships or task links.

## Fallback path

The fallback is the existing behavior:

1. deep-copy one recipe per estimator;
2. fit and transform each copy independently;
3. stack outputs as today;
4. apply `recipe.output`.

The new executor can start by dispatching to fallback unless all processors in a
prefix advertise safe ensemble behavior. This lets the migration be incremental
and keeps user-defined processors correct without modification.

## Migration plan

Quick win:

- Add private `EnsembleContext`, `EnsembleTable`, and fallback executor types.
- Implement ensemble-aware `Sequential` for shared deterministic prefixes.
- Implement grouped `Choice` for `Identity`/`Power` branch grouping.
- Implement grouped `FeaturePermute` and `CategoryShuffle` transforms only where
  parity is straightforward.
- Use the new executor from `ICLModel.forward` and `ICLModel.fit` behind the
  existing `num_estimators` argument.

Medium effort:

- Add stable processor fingerprints for built-in processors.
- Add explicit `fit_state_key` implementations for `StandardScale`, `Power`,
  `Quantile`, categorical aligners, datetime encoders, and deterministic text
  encoders when introduced.
- Share related-table deterministic prefixes per table key.
- Add profiler-backed benchmarks to show how much time is saved by avoiding
  repeated fit work versus faster individual processors.

Long-term:

- Let more processors operate directly on a leading estimator/member dimension.
- Support grouped materialization for multiple independent stochastic axes.
- Add custom kernels only if PyTorch primitives cannot approach the measured
  speed-of-light after redundant fits are removed.

## Correctness invariants

- Learned state must depend only on context/train data, never query data.
- Reused state must be equivalent to fitting the same processor on the same
  input in the current implementation.
- Stochastic processors must preserve user-visible RNG behavior. If preserving
  exact draw order conflicts with grouping, exact parity wins.
- A processor that mutates fitted state during transform is not shareable unless
  it provides explicit ensemble-safe semantics.
- Fitted state is scoped by table identity for related tables.
- Query transforms may reuse fitted context state, but may not update it.
- Device and dtype must be preserved; no host-device synchronization should be
  introduced into hot paths.

## Parity and benchmark plan

Parity should be proven before enabling the executor by default:

- strict TabICLv2 parity for 1 estimator and multiple estimators;
- fixed-generator tests where `Choice`, `FeaturePermute`, and `CategoryShuffle`
  produce the same logical member outputs as the current cloned-recipe loop;
- regression and classification target tests, including target inverse/output
  reduction;
- related-table tests proving table-scoped state is not shared across different
  table schemas or task links;
- user-defined processor tests proving fallback preserves current behavior.

Performance should be measured with synchronization-aware CUDA timing:

- largest TabICL workload: 40k context rows, 10k query rows, 100 features,
  8 estimators split as 4 power norm and 4 no norm;
- report current preprocessing runtime, grouped executor runtime, median/p95,
  peak GPU memory, kernel time, transfer time, and synchronization overhead;
- target movement from about 0.46-0.47 s toward about 0.19-0.20 s on the tested
  GPU with strict parity;
- RFM microbenchmark with multiple related tables and an expensive deterministic
  placeholder/text encoder to quantify table-scoped reuse.

## Recommendation

The smallest maintainable first implementation is an internal ensemble-aware
executor with opt-in processor hooks and a conservative fallback. Start with
`Sequential`, `Choice`, `FeaturePermute`, and `CategoryShuffle`, because those
define where TabICL's deterministic prefix can remain shared and where member
specificity begins. Add `fit_state_key` only to processors whose state is easy to
fingerprint and immutable after fit. This should close most of the TabICL
preprocessing gap without introducing a processor-specific `Power` cache or
requiring every processor to understand ensembling immediately.
