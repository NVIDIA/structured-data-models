# Ensemble-Aware Processing: Implementation Design

This document defines the components and interactions for shared and vectorized Recipe execution across ensemble members. The [problem definition](ensemble_aware_processing_problem.md) contains motivation, correctness constraints, scope, and performance goals.

## Recipe

`Recipe` is the public owner of the feature, target, related-table, and output processing paths:

```python
class Recipe(torch.nn.Module):
    def fit_transform(
        self,
        features: TableTensor,
        target: TableTensor,
        related_tables: RelatedTables | None = None,
        *,
        num_members: int,
        generator: torch.Generator | None = None,
    ) -> tuple[
        EnsembleTable,
        EnsembleTable,
        EnsembleRelatedTables | None,
    ]: ...

    def transform(
        self,
        features: TableTensor,
        related_tables: RelatedTables | None = None,
    ) -> tuple[EnsembleTable, EnsembleRelatedTables | None]: ...

    def transform_output(self, outputs: Sequence[TableTensor]) -> TableTensor: ...
```

`fit_transform` creates one shared `EnsembleTable` per logical table, invokes the root Processors through `fit_transform_ensemble`, builds all fitted Processor trees temporarily, and installs them atomically. `transform` reuses those trees and their member decisions. Each related table has its own fitted tree; fitted state is not shared across tables.

`transform_output` applies the member-aligned fitted target inverse for regression or class alignment for classification, then runs the output pipeline. TabICLv2 does not perform target inversion manually.

## EnsembleTable

```python
class EnsembleTable:
    groups: tuple[TableTensor, ...]  # each [V_g, ..., R, C]
    member_to_variant: tuple[tuple[int, int], ...]  # E -> (group, variant)

    @classmethod
    def from_shared(
        cls,
        table: TableTensor,
        *,
        num_members: int,
    ) -> EnsembleTable: ...

    @classmethod
    def pack(
        cls,
        variants: Sequence[TableTensor],
        member_to_input_variant: tuple[int, ...],
    ) -> EnsembleTable: ...

    def __getitem__(self, member_id: int) -> TableTensor:
        group, variant = self.member_to_variant[member_id]
        return self.groups[group][variant]

    def with_groups(self, groups: tuple[TableTensor, ...]) -> EnsembleTable: ...
```

Each group contains unique variants with compatible shape, schema, stypes, dtypes, and categorical metadata. `member_to_variant` maps the stable member position to `(group, variant)`; equal references represent proven sharing. `pack` groups compatible unique results and creates separate groups for incompatible results. It never merges variants by comparing tensor contents.

`EnsembleRelatedTables` maps table names to `EnsembleTable`; relationships and task links remain shared graph metadata.

## Processor Contracts

```python
class EnsembleProcessor(Processor):
    def fit_ensemble(self, table: EnsembleTable, *, context: EnsembleFitContext) -> Self: ...
    def transform_ensemble(self, table: EnsembleTable) -> EnsembleTable: ...
    def fit_transform_ensemble(self, table: EnsembleTable, *, context: EnsembleFitContext) -> EnsembleTable: ...

class VariableSchemaBatchMixin:
    def fit_batch(self, batch: TableTensor, *, generator: torch.Generator | None = None) -> Self: ...
    def transform_batch(self, batch: TableTensor) -> tuple[TableTensor, ...]: ...
```

`EnsembleProcessor` retains the inherited `TableTensor → TableTensor` API; the bridge to its ensemble methods is an implementation detail outside this initial specification. Invertible implementations also expose `inverse_transform_ensemble`. A normal `Processor` operates independently on every leading variant position and retains fitted state such as `[V_g,1,C]`; member-routing, composite, or stochastic operations implement `EnsembleProcessor` directly. A `VariableSchemaBatchMixin` remains a normal `Processor`, exposes an explicit batch path when fitted output schemas may differ by batch position, and preserves the single-table `TableTensor` return type.

## Processor Adapter

```python
class _EnsembleProcessorAdapter(EnsembleProcessor):
    def fit_transform_ensemble(
        self,
        table: EnsembleTable,
        *,
        context: EnsembleFitContext,
    ) -> EnsembleTable:
        self.processors = ModuleList(
            copy.deepcopy(self.template) for _ in table.groups
        )
        if isinstance(self.template, VariableSchemaBatchMixin):
            ...

        else...


def as_ensemble_processor(
    processor: Processor,
) -> EnsembleProcessor:
    if isinstance(processor, EnsembleProcessor):
        return processor
    return _EnsembleProcessorAdapter(processor)
```

`_EnsembleProcessorAdapter` adapts a normal Processor to the ensemble contract and owns one fitted instance per input group. For a `VariableSchemaBatchMixin`, its private variable-schema path requires one output per batch position and passes those outputs with the composed member mapping to `pack`; `DropConstantColumns` uses this path so mask calculation remains vectorized. Query transform and inverse transform select the same path and apply the same instances to groups in the same order. All composite children are normalized through `as_ensemble_processor`, so downstream components use one interface.

## Variant-Producing and Composite Processors

- Member-specific randomness is defined by stable `(member_id, table_scope, processor_path)` streams and remains independent of physical variant grouping and model randomness. Decisions are sampled during fit, stored by the Processor, and reused by transform and inverse transform.
- `Choice` stores one option per member, computes each unique selected branch once, and calls `EnsembleTable.pack` on the branch results in original member order. Round-robin selects `member_id % num_options`; random selection uses the member stream.
- `ShuffleColumns` and `ShuffleCategories` store member-specific mappings, compute unique results, and call `pack`.
- `Sequential`, `StypeDispatch`, and `TaskDispatch` pass `EnsembleTable` recursively through normalized children.
- Only `ReduceEstimators` may aggregate the member dimension.

The producer owns the semantic member-to-result mapping. `pack` owns physical grouping. The group-preserving adapter does not repack; variant producers and the variable-schema adapter do.

## Lazy Member Fitting Extension

The proposed extension keeps the existing `EnsembleTable` and fitted Processor tree but permits a branch to contain only a subset of the total ensemble. No separate lazy container is required.

```python
class EnsembleTable:
    groups: tuple[TableTensor, ...]  # each [V_g, ..., R, C]
    member_to_location: Tensor  # [E, 2], (group, position)
```

`member_to_location` is a clearer proposed name for `member_to_variant`: it describes the physical location of each member's current processed representation. Its length always equals the total ensemble size. `(-1, -1)` marks a member that is inactive in the current branch. Compatible representations remain in one group with a leading representation dimension; separate groups are required only for incompatible schemas or metadata.

An ensemble-aware Processor may fit all active members or a requested subset. For each requested member, it resolves the input location and fits that data-dependent representation only if no fitted state exists. Every active member pointing to the same location shares that fitted state and is marked fitted at the same time. Sharing follows execution provenance; tensor values are never compared.

```python
class EnsembleProcessor(Processor):
    member_to_fitted_state: Tensor  # [E], -1 if not fitted

    def fit_transform_ensemble(
        self,
        table: EnsembleTable,
        *,
        context: EnsembleFitContext,
        members: Sequence[int] | None = None,
    ) -> EnsembleTable: ...
```

- Data-dependent Processors such as `Standardize` or `PowerTransform` fit at most once per active input location and reuse that state for query transformation.
- Routing-only Processors such as `Choice`, `ShuffleColumns`, and `ShuffleCategories` create their member decisions once from ensemble size, schema, and RNG. They do not need one fitted Processor instance per member. A selected `Choice` child may still be data-dependent and is fitted lazily for the members reaching that branch.
- `transform_ensemble` remains read-only and never triggers fitting. Adding members later requires another explicit context `fit_transform_ensemble(..., members=...)` call.
- Branches retain a full-length member mapping; members outside the branch stay inactive, so branch results can be combined without reconstructing global member positions.

## Tasks

| Task                          | Scope                                                                                                                                                                 | Owner                                                                           |
| ----------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| Recipe and output integration | Update `Recipe.fit_transform`/`transform`, atomic state, related tables, and `transform_output` including target inverse and class alignment.                         | TBD                                                                             |
| Ensemble container            | Implement `EnsembleTable`, `pack`, member lookup, and related-table container.                                                                                        | Ramona Bendias DE                                                               |
| Custom ensemble nodes         | Add `EnsembleProcessor` behavior for `Choice`, `Sequential`, `StypeDispatch`, `TaskDispatch`, `ShuffleColumns`, `ShuffleCategories`, and sampled `QuantileTransform`. | Jana Gagacheva (`Choice`/`Sequential`/`StypeDispatch`); remaining ownership TBD |
| Processor adapter             | Implement `_EnsembleProcessorAdapter` and `VariableSchemaBatchMixin` with one fitted instance per input group and transform/inverse reuse.                            | TBD                                                                             |
| Vectorized leaf Processors    | Update normal Processors to accept independent leading variant dimensions `[V_g,...,R,C]`.                                                                            | Ramona Bendias DE                                                               |
| Validation                    | Cover member RNG, schema grouping, context/query state reuse, target inverse, raw member parity, and CUDA performance.                                                | TBD                                                                             |

## Open Questions

- Serialization of dynamically fitted group Processors and member decisions.
- Whether `member_to_variant` should be renamed to `member_to_location` publicly or retained for API compatibility when lazy member fitting is implemented.
