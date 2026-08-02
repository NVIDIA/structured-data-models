# Ensemble-Aware Processing: Implementation Design

This document defines the components and interactions for shared and vectorized Recipe execution across ensemble members. The [problem definition](ensemble_aware_processing_problem.md) contains motivation, correctness constraints, scope, and performance goals.

## Recipe (note: exact API is not decided yet - needs input from Matthias)

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
    _packed_representations: tuple[TableTensor, ...]
    _member_locations: tuple[tuple[int, int], ...]

    def __init__(self, table: TableTensor, *, num_members: int) -> None: ...

    @classmethod
    def from_representations(
        cls,
        representations: Sequence[TableTensor],
        member_representation_ids: tuple[int, ...],
    ) -> Self: ...

    def iter_packed_representations(self) -> Iterator[TableTensor]: ...

    def repack(self, representations: Sequence[TableTensor]) -> Self: ...
```

Each data representation is a `TableTensor` whose leading dimension holds unique batches with compatible shape, schema, stypes, dtypes, and categorical metadata. `_member_locations` maps the stable member position to `(representation_id, batch_id)`; equal references represent proven sharing. `from_representations` packs compatible unique results into the same data representation and creates separate representations for incompatible results. It never merges representations by comparing tensor contents.

`EnsembleRelatedTables` maps table names to `EnsembleTable`; relationships and task links remain shared graph metadata.

Construction initially maps every member to one shared representation. Processors operate on that representation once and only create additional representations when routing, randomness, or incompatible output schemas require a split. Compatible distinct representations remain vectorized along the leading batch dimension. `repack` accepts one result per existing `(representation_id, batch_id)` in `iter_packed_representations` order, preserves the member mapping, and packs compatible results again. This is lazy materialization: tensor operations remain eager, but member data is neither copied nor separately processed before its execution path diverges.

## Processor Contracts

```python
class EnsembleTableProcessor(Processor):
    def fit_ensemble(self, table: EnsembleTable, *, context: EnsembleFitContext) -> Self: ...
    def transform_ensemble(self, table: EnsembleTable) -> EnsembleTable: ...
    def fit_transform_ensemble(self, table: EnsembleTable, *, context: EnsembleFitContext) -> EnsembleTable: ...
```

`EnsembleTableProcessor` retains the inherited `TableTensor → TableTensor` API; the base class bridges that API through an `EnsembleTable` with one member. Invertible implementations also expose `inverse_transform_ensemble`.

A normal `Processor` consumes and returns one `TableTensor`. It operates independently on every leading batch position within a packed representation and may retain fitted state such as `[B,1,C]`. It may change the schema only when every leading position produces the same output schema, so the result remains one `TableTensor`.

An operation implements `EnsembleTableProcessor` when it must inspect or change the `EnsembleTable` structure. This includes member routing, composites, member-specific randomness, and data-dependent schema splits. There is no separate `VariableSchemaProcessor`, mixin, or capability flag. For example, `DropConstantColumns` computes masks vectorized as `[B,C]`, stores the fitted columns per packed position, creates one output per position, and calls `repack`; query transformation reuses those columns rather than recomputing them. `AlignCategories` follows the same contract when fitted output schemas differ.

## Processor Adapter

```python
class _EnsembleTableProcessorAdapter(EnsembleTableProcessor):
    def fit_transform_ensemble(
        self,
        table: EnsembleTable,
        *,
        context: EnsembleFitContext,
    ) -> EnsembleTable:
        self.processors = ModuleList(
            copy.deepcopy(self.template) for _ in table._packed_representations
        )
        ...


def as_ensemble_table_processor(
    processor: Processor,
) -> EnsembleTableProcessor:
    if isinstance(processor, EnsembleTableProcessor):
        return processor
    return _EnsembleTableProcessorAdapter(processor)
```

`_EnsembleTableProcessorAdapter` adapts a normal Processor to the ensemble contract and owns one fitted instance per packed data representation. It applies the Processor to the complete leading batch and replaces the packed representation without changing its member mapping. Query transform and inverse transform apply the same fitted instances to packed representations in the same order. The adapter never handles schema splits; operations that may produce a different schema per leading position implement `EnsembleTableProcessor` directly. All composite children are normalized through `as_ensemble_table_processor`, so downstream components use one interface. The adapter is private execution machinery, not a Processor category users select.

## Representation-Producing and Composite Processors

- Member-specific randomness is defined by stable `(member_id, table_scope, processor_path)` streams and remains independent of physical packing of data representations and model randomness. Decisions are sampled during fit, stored by the Processor, and reused by transform and inverse transform.
- `Choice` stores one option per member, computes each unique selected branch once, and calls `EnsembleTable.pack` / `from_representations` on the branch results in original member order. Round-robin selects `member_id % num_options`; random selection uses the member stream.
- `ShuffleColumns` and `ShuffleCategories` store member-specific mappings, compute unique results, and call `pack`.
- `DropConstantColumns` and `AlignCategories` compute data-dependent state vectorized per packed representation and call `repack` when leading positions produce incompatible schemas.
- `Sequential`, `StypeDispatch`, and `TaskDispatch` pass `EnsembleTable` recursively through normalized children.
- Only `ReduceEstimators` may aggregate the member dimension.

The producer owns the semantic member-to-result mapping. `pack` / `from_representations` owns physical packing into data representations and `batch_id` assignment. The representation-preserving adapter does not repack; `EnsembleTableProcessor` implementations that produce new routes or schemas do.

## Lazy Member Fitting Extension (Extension - not needed in v0)

The proposed extension keeps the existing `EnsembleTable` and fitted Processor tree but permits a branch to contain only a subset of the total ensemble. No separate lazy container is required.

```python
class EnsembleTable:
    _packed_representations: tuple[TableTensor, ...]  # each [B, ..., R, C]
    _member_locations: Tensor  # [E, 2], (representation_id, batch_id)
```

`_member_locations` describes the physical location of each member's current processed data representation. Its length always equals the total ensemble size. `(-1, -1)` marks a member that is inactive in the current branch. Compatible results remain in one packed data representation with a leading batch dimension; separate representations are required only for incompatible schemas or metadata.

An `EnsembleTableProcessor` may fit all active members or a requested subset. For each requested member, it resolves the input location and fits that data-dependent representation only if no fitted state exists. Every active member pointing to the same `(representation_id, batch_id)` shares that fitted state and is marked fitted at the same time. Sharing follows execution provenance; tensor values are never compared.

```python
class EnsembleTableProcessor(Processor):
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

| Task                          | Scope                                                                                                                                                 | Owner                                                                           |
| ----------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| Recipe and output integration | Update `Recipe.fit_transform`/`transform`, atomic state, related tables, and `transform_output` including target inverse and class alignment.         | TBD                                                                             |
| Ensemble container            | Implement `EnsembleTable`, packing into data representations, member lookup via `(representation_id, batch_id)`, and related-table container.         | Ramona Bendias DE                                                               |
| Collection-aware processors   | Add `EnsembleTableProcessor` behavior for routing, composites, stochastic operations, and data-dependent schema splits such as `DropConstantColumns`. | Jana Gagacheva (`Choice`/`Sequential`/`StypeDispatch`); remaining ownership TBD |
| Processor adapter             | Implement `_EnsembleTableProcessorAdapter` with one fitted instance per packed representation and transform/inverse reuse.                            | TBD                                                                             |
| Vectorized leaf Processors    | Update normal Processors to accept independent leading batch dimensions `[B,...,R,C]` within a data representation.                                   | Ramona Bendias DE                                                               |
| Validation                    | Cover member RNG, representation packing by schema, context/query state reuse, target inverse, raw member parity, and CUDA performance.               | TBD                                                                             |

## Open Questions

- Serialization of dynamically fitted per-representation Processors and member decisions.
- Whether public accessors should expose `_member_locations` as `(representation_id, batch_id)` only, or also retain a transitional alias for older `member_to_variant` / group naming when lazy member fitting is implemented.
