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

`fit_transform` creates one shared `EnsembleTable` per logical table, builds all fitted Processor trees temporarily, and installs them atomically. `transform` reuses those trees and their member decisions. Each related table has its own fitted tree; fitted state is not shared across tables.

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
class Processor(torch.nn.Module):
    # One schema-compatible TableTensor [V_g, ..., R, C] -> TableTensor
    ...

class EnsembleProcessor(torch.nn.Module):
    # The complete EnsembleTable -> EnsembleTable
    def fit_transform(
        self,
        table: EnsembleTable,
        *,
        context: EnsembleFitContext,
    ) -> EnsembleTable: ...
    def transform(self, table: EnsembleTable) -> EnsembleTable: ...
    def inverse_transform(self, table: EnsembleTable) -> EnsembleTable: ...
```

A normal `Processor` operates independently on every leading variant position. Fit reduces only over the row axis `-2`, so fitted state retains the variant dimension, for example `[V_g,1,C]`. Its output remains compatible within the input group. Structural, composite, or stochastic operations implement `EnsembleProcessor` directly.

## Normal Processor Adapter

```python
class _VariantGroupProcessor(EnsembleProcessor):
    def fit_transform(
        self,
        table: EnsembleTable,
        *,
        context: EnsembleFitContext,
    ) -> EnsembleTable:
        self.processors = ModuleList(
            copy.deepcopy(self.template) for _ in table.groups
        )
        groups = tuple(
            processor.fit_transform(
                group,
                generator=context.generator,
            )
            for processor, group in zip(self.processors, table.groups)
        )
        return table.with_groups(groups)

def as_ensemble_processor(
    processor: Processor | EnsembleProcessor,
) -> EnsembleProcessor:
    if isinstance(processor, EnsembleProcessor):
        return processor
    return _VariantGroupProcessor(processor)
```

`_VariantGroupProcessor` owns one fitted normal Processor instance per input group. Query transform and inverse transform apply the same instances to groups in the same order. All composite children are normalized through `as_ensemble_processor`, so downstream components use one interface.

## Variant-Producing and Composite Processors

- Member-specific randomness is defined by stable `(member_id, table_scope, processor_path)` streams and remains independent of physical variant grouping and model randomness. Decisions are sampled during fit, stored by the Processor, and reused by transform and inverse transform.
- `Choice` stores one option per member, computes each unique selected branch once, and calls `EnsembleTable.pack` on the branch results in original member order. Round-robin selects `member_id % num_options`; random selection uses the member stream.
- `FeaturePermute` and `CategoryShuffle` store member-specific mappings, compute unique results, and call `pack`.
- `Sequential`, `StypeDispatch`, and `TaskDispatch` pass `EnsembleTable` recursively through normalized children.
- Only `EnsembleReduce` may aggregate the member dimension.

The producer owns the semantic member-to-result mapping. `pack` owns physical grouping. The following `_VariantGroupProcessor` consumes the groups without repacking them.

## Example Flow

For `ConstantFilter → StandardScale → Choice(Identity, Power) → FeaturePermute → Clip` with eight members:

1. Recipe creates one group `[V_0=1,N,D]` referenced by all eight members.
2. The deterministic prefix fits once on that shared group.
3. `Choice` computes one Identity and one Power result; `pack` forms `[V_0=2,N,D]` and maps the eight members to those two variants.
4. Following normal Processors operate directly on each packed group through `_VariantGroupProcessor`.
5. `FeaturePermute` creates new unique variants and calls `pack`; later Processors consume the resulting groups.
6. Query transform repeats the stored branch and grouping decisions. `table[member_id]` provides the corresponding model input.

## Open Questions

- Fit-dependent schema splits within one group: a custom `EnsembleProcessor` or a per-variant fitting adapter followed by `pack`.
- `Sequential` contract: ensemble-only composition, or also direct execution as a normal Processor; Recipe must retain one canonical execution path.
- Serialization of dynamically fitted group Processors and member decisions.
