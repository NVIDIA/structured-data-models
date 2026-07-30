# Ensemble-Aware Processing: Implementation Design

The implementation keeps one logical member order while storing only proven-distinct table variants. Work remains shared until a Processor introduces a different decision or schema.

```text
TableTensor
    │ Recipe.fit_transform(num_members=E)
    ▼
EnsembleTable: one shared variant ── deterministic/vectorized Processors ──┐
    │ Choice or member-specific mapping                                   │
    ├── branch/group A ── compatible variants execute together            │
    └── branch/group B ── incompatible schemas execute separately         │
                                                                         ▼
Model scheduler: compatible member groups or singleton fallback
                                                                         │
member-aligned outputs ── TargetDecode ── ReduceEstimators ──────────────┘
```

## Recipe owns the graph

```python
class Recipe(torch.nn.Module):
    def fit_transform(
        self,
        features: TableTensor,  # [R, C] with variable-schema steps
        target: TableTensor,  # [R, 1] with variable-schema steps
        related_tables: RelatedTables | None = None,
        *,
        num_members: int,
        generator: torch.Generator | None = None,
    ) -> tuple[EnsembleTable, EnsembleTable, EnsembleRelatedTables | None]: ...

    def transform(
        self,
        features: TableTensor,
        related_tables: RelatedTables | None = None,
    ) -> tuple[EnsembleTable, EnsembleRelatedTables | None]: ...

    def transform_output(self, outputs: Sequence[TableTensor]) -> TableTensor: ...
```

`fit_transform` creates a shared `EnsembleTable` for each logical input table, builds temporary fitted Processor trees, and installs the complete plan only after every path succeeds. `transform` reuses exactly those trees and member decisions. Each related table gets its own feature tree and RNG scope; no fitted state crosses table boundaries.

## Ensemble representation

```python
@dataclass(frozen=True)
class EnsembleTable:
    groups: tuple[TableTensor, ...]  # each [V_g, R, C] in version 1
    member_to_variant: tuple[tuple[int, int], ...]  # member -> (group, variant)
```

- A group contains unique variants with compatible shape, schema, stypes, dtypes, device, and categorical metadata. Version 1 accepts one unbatched logical table per Recipe call when variable-schema processors are present; the leading group dimension is reserved for proven ensemble variants.
- `member_to_variant` preserves stable member order and proven sharing.
- `from_shared(table, num_members=E)` stores one view referenced by all members.
- `pack(variants, member_to_input_variant)` groups compatible results without comparing tensor values.
- `materialize(member_ids)` creates a physical leading member dimension only at a consumer boundary.
- Singleton groups and singleton materializations use `unsqueeze` views; true multi-variant groups use stacking.
- `EnsembleRelatedTables` maps table names to `EnsembleTable` while sharing relationships and task links.

## Processor contracts

```python
class Processor(torch.nn.Module):
    supports_leading_variants: ClassVar[bool] = False
    member_specific_fit: ClassVar[bool] = False


class EnsembleProcessor(Processor):
    def fit_transform_ensemble(
        self,
        table: EnsembleTable,
        *,
        context: EnsembleFitContext,
    ) -> EnsembleTable: ...

    def transform_ensemble(self, table: EnsembleTable) -> EnsembleTable: ...
```

`EnsembleProcessor` remains a normal `Processor`: scalar `TableTensor` calls are represented internally as a one-member `EnsembleTable` and return a scalar `TableTensor`.

`as_ensemble_processor` normalizes every node to one execution contract:

- A leaf with `supports_leading_variants=True` is wrapped by `_EnsembleProcessorAdapter`, fitted once per compatible group, and processes its leading variants independently.
- A `VariableSchemaBatchMixin` implements `fit_batch`/`transform_batch` on `[V, R, C]`, returning one table per leading variant; the adapter then repacks compatible schemas. `DropConstantColumns` and `AlignCategories` use this path.
- `member_specific_fit=True` creates one fitted child per member when fitted state must remain member-local and no specialized structural implementation exists.
- A leaf satisfying none of these contracts raises `TypeError`; there is no silent scalar fallback.
- The adapter rejects row-count changes.

## Structural and stochastic nodes

- `Sequential` passes the current `EnsembleTable` through normalized children.
- `StypeDispatch` routes each semantic type as an `EnsembleTable`, executes active routes, and recombines member-aligned results.
- `TaskDispatch` is resolved from the fitted target before execution and uses one consistent task branch.
- `Choice(selection="round_robin")` selects `member_id % len(options)`; random choice uses stable member streams. Each selected branch is computed once for its selected subset and merged back in member order.
- `ShuffleColumns` and `ShuffleCategories` store member-specific mappings, deduplicate equal mappings, and vectorize compatible distinct results.
- An optional `EnsemblePlanner` supplies exact model-family reference plans without hard-coding them into generic Processors. TabICLv2 uses it for paired normalization plus Latin feature and shifted class permutations.

## RNG ownership

`EnsembleFitContext` derives deterministic streams from the root seed, stable member ID, logical table scope, and Processor path. Grouping or repacking therefore cannot change a member's decision. Fit stores every decision; transform and inverse transform never resample it. On automatic CUDA OOM retry, model execution restores the generator state before running the sequential schedule.

## Model scheduling

`ICLModel.forward` and `fit` accept `ensemble_mode="parallel" | "sequential" | "auto"`. A model core opts into a leading ensemble dimension with `supports_vectorized_ensemble=True`; otherwise every schedule uses singleton model calls. If grouped calls consume a generator, the model must additionally opt into `supports_vectorized_ensemble_rng=True` and guarantee member-wise RNG equivalence.

- `parallel` groups members with compatible feature, target, and related-table execution signatures, materializes each group, and invokes a vectorized model core.
- `sequential` invokes the same fitted Recipe plan but materializes and executes one member at a time.
- `auto` attempts parallel execution and retries sequentially only after a CUDA OOM, releasing cached GPU memory first.
- A model with `supports_vectorized_ensemble=False` always receives scalar members. KumoRFM currently uses this mode: its preprocessing is shared, while its relational model core remains sequential.

## Output semantics

`Recipe.transform_output` keeps outputs in stable member order. `TargetDecode` explicitly maps regression values through each member's fitted target inverse or aligns classification logits to canonical classes and is required before `ReduceEstimators`. Compatible stateless steps before reduction process the stacked estimator dimension directly. `ReduceEstimators` must be a direct output step, may occur at most once, and is the only member aggregation boundary. Fitted output nodes, ambiguous nested reductions, feature/target reductions, member-specific work after reduction, and decode after reduction fail during Recipe validation.

## Boundaries and follow-up

- Supported now: row-preserving processing, variable column schemas, table-local related state, strict TabICLv2 reference planning, deterministic materialization, direct and cached model execution, and CUDA OOM fallback.
- Not supported: row-changing Processors, cross-table fitted-state reuse, cross-device groups, multi-GPU scheduling, or a vectorized KumoRFM model core.
- Serialization of dynamically fitted group Processors and member decisions needs an explicit stable format before it becomes a public persistence contract.
- `StypeDispatch` inverse transformation currently requires one active route for the complete ensemble.

An alternative was to keep fitted Recipe copies and add a cache above them. That can reuse selected states but retains duplicate execution trees, does not carry intermediate provenance, and cannot vectorize later distinct variants directly. The Processor-level ensemble contract is preferred because sharing, splitting, fitting, transform, and inverse transform use one representation and one execution path.
