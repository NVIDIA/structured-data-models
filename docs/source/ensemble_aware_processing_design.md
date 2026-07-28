# Ensemble-Aware Processing: Implementation Design

This document defines the recommended solution for shared and vectorized Recipe execution across ensemble members. The [problem definition](ensemble_aware_processing_problem.md) contains the motivation, mathematical formulation, and speed of light. API compatibility is not required.

## Decisions

- `Recipe` is the only public ensemble-processing entry point and processes features, target, and related tables together.
- `Recipe` is fitted in place; refitting replaces all fitted state only after every table has fitted successfully.
- `EnsembleTable` stores unique variants in schema-compatible tensor groups `[V_g,...,R,C]` and maps each stable member position to one variant.
- A private `_VariantGroupProcessor` owns one fitted Processor instance per variant group; `EnsembleTable` contains no fitted state.
- Every normal Processor supports independent leading dimensions and operates directly on one variant group. There is no per-member fallback.
- Structural, composite, and stochastic Processors override the internal ensemble execution.
- Processors share variants only through execution provenance, never through tensor comparison or hashing.
- `Recipe.transform_output` owns regression target inversion and classification alignment; TabICLv2 no longer performs target inversion manually.
- Version 1 is row-preserving, table-local, immutable during `transform`, and limited to one device per execution.

## Recipe API

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
    ) -> tuple[EnsembleTable, EnsembleTable, EnsembleRelatedTables | None]: ...

    def transform(
        self,
        features: TableTensor,
        related_tables: RelatedTables | None = None,
    ) -> tuple[EnsembleTable, EnsembleRelatedTables | None]: ...

    def transform_output(self, outputs: Sequence[TableTensor]) -> TableTensor: ...
```

The model passes `num_estimators` as `num_members`. `fit_transform` builds feature, target, and related-table state temporarily and installs it atomically. `transform` uses only this state. Recipe is copied neither per member nor externally per table.

## EnsembleTable

```python
class EnsembleTable:
    groups: tuple[TableTensor, ...]  # each [V_g, ..., R, C]
    member_to_variant: tuple[tuple[int, int], ...]  # E -> (group, variant)

    @classmethod
    def from_shared(cls, table: TableTensor, *, num_members: int) -> EnsembleTable: ...

    @classmethod
    def pack(
        cls,
        variants: Sequence[TableTensor],
        member_to_input_variant: tuple[int, ...],
    ) -> EnsembleTable: ...

    @property
    def num_members(self) -> int: ...

    def __getitem__(self, member_id: int) -> TableTensor:
        group, variant = self.member_to_variant[member_id]
        return self.groups[group][variant]

    def with_groups(self, groups: tuple[TableTensor, ...]) -> EnsembleTable: ...
```

- `member_to_variant` has length `E`; its position is the member ID and its value `(group, variant)` addresses one leading tensor position.
- `from_shared(table, num_members=4)` creates `groups=(table[None],)` and `member_to_variant=((0,0),)*4`.
- Equal references mean proven sharing. An Identity/Power Choice may produce `((0,0),(0,1),(0,0),(0,1))`.
- References are local and contain no execution history. Branch selection and fitted-state alignment belong to the fitted Processors.
- Every referenced variant exists and is used by at least one member. Content-equal tensors are not merged without shared provenance.
- Within one group, shape, schema, stypes, block dtypes, and categorical metadata match. Groups may differ in these properties but remain on one device in version 1.
- `EnsembleRelatedTables` maps each table name to an `EnsembleTable`; relationships and task links remain shared metadata.

## Normal Processor

A normal `Processor` knows neither members nor variant groups. Its tensor blocks support `[R,C]`, `[V_g,R,C]`, and additional leading dimensions. In the worst case, the sum of all `V_g` equals `E`; with sharing it is smaller.

The contract is strict:

- Every leading position is processed independently and never aggregated with another.
- Rows are axis `-2` and columns are axis `-1`; fit reduces only over `-2`.
- Fitted state preserves leading dimensions, for example `mean: [V_g,1,C]`.
- `transform` and `inverse_transform` do not mutate fitted state.
- One invocation returns a dense output with compatible shape and identical metadata for every leading position.
- Member-specific randomness, cross-variant side effects, and variant-dependent incompatible output schemas are not allowed.

An external Processor satisfies this contract or overrides internal ensemble execution; otherwise it is rejected. Unchecked callables are not wrapped automatically on the ensemble path.

## Variant Groups and Packing

```python
class _VariantGroupProcessor(torch.nn.Module):
    def fit_transform(self, table: EnsembleTable, *, context: EnsembleFitContext) -> EnsembleTable:
        self.processors = ModuleList(copy.deepcopy(self.template) for _ in table.groups)
        groups = tuple(processor.fit_transform(group, generator=context.generator) for processor, group in zip(self.processors, table.groups))
        return table.with_groups(groups)
```

`_VariantGroupProcessor` stores group order and one Processor instance per group. `transform` and inverse transform apply these instances in the same order; query may have a different row count. A Processor that creates variants calls `EnsembleTable.pack(...)`. `pack` stacks compatible unique results, forms separate groups for incompatible schemas, and creates `member_to_variant`. The producing Processor defines only the semantic mapping; the following wrapper does not repack.

Normal built-ins such as `Identity`, `Clip`, `ToNumerical`, `EncodeDatetime`, `SoftmaxTemperature`, `MeanImpute`, `StandardScale`, `QuantileClip`, `SigmaClip`, `Power`, and `CategoricalImpute` are adapted to this contract.

## Processors with Custom Ensemble Semantics

- `ConstantFilter` and `CategoricalAlign` need semantics for differing fitted output schemas when one input group contains multiple variants.
- `Choice` stores one option per member, computes each unique branch variant once, and calls `pack` for branch outputs in original member order. Round-robin uses `member_id % num_options`.
- `FeaturePermute` and `CategoryShuffle` store member-specific mappings, create unique outputs, and call `pack` for compatible results.
- `Quantile` owns its RNG and variant semantics when sampling.
- `Sequential`, `StypeDispatch`, and `TaskDispatch` delegate recursively to their children; Recipe contains no Processor-specific branches.
- Only `EnsembleReduce` may aggregate members.

## Minimal Execution

For eight members and `ConstantFilter → StandardScale → Choice(Identity, Power) → FeaturePermute → Clip`:

1. Recipe starts with one group `[V_0=1,N,D]` and `member_to_variant=((0,0),)*8`.
2. `ConstantFilter` and `StandardScale` each fit the shared prefix once.
3. `Choice` creates two branch outputs for four Identity and four Power members; `pack` forms one group `[V_0=2,N,D]`, and `Power` fits once.
4. The following normal Processor wrapper owns one instance and processes `[V_0=2,N,D]` directly.
5. `FeaturePermute` creates only unique branch/permutation combinations and packs compatible outputs; the `Clip` wrapper processes each resulting group directly.
6. Query transform repeats the same variant path using stored states and decisions.

`Recipe.transform_output` applies the fitted target inverse for regression and class alignment for classification per member; TabICLv2 has no manual special path. Materialization in member order occurs only before model or output operations requiring a dense tensor. For RFM, the task table, target, and every related table own separate fitted Processor trees. Sharing occurs between members within one table, not between tables.

## Implementation and Acceptance

1. Choose the RNG contract: current-main draw order or stable streams per `(member, table_scope, processor_path)`.
2. Implement `EnsembleTable.pack`, `_VariantGroupProcessor`, and atomic Recipe fit.
3. Assign every built-in to the normal or custom ensemble contract; provide no silent fallback.
4. Add `Choice`, permutations, dispatch, target inversion, output alignment, and RFM.
5. Test parity for context, query, fitted state, raw member outputs, RNG, and dtype-specific tolerances.
6. Profile CUDA stacking copies, kernel launches, synchronization, and peak memory.

Target workload: 40k context rows, 10k query rows, 100 features, and eight members with four Power and four Identity variants. Strict-parity Recipe processing should move from approximately `0.46–0.47 s` toward `0.19–0.20 s`. Measure preprocessing and the complete model path separately until a shared end-to-end target is defined.

## Open Questions

- RNG: legacy draw parity or separate member-scope streams.
- Serialization of fitted Processor states and member-specific decisions.
- Fit-dependent schema splits within one group: a custom ensemble Processor or a per-variant fitting wrapper followed by `pack`, especially for `ConstantFilter` and `CategoricalAlign`.
- `Sequential` contract: ensemble-aware composition over `EnsembleTable` only, or also direct execution as a normal Processor on one `TableTensor`; Recipe keeps exactly one canonical execution path either way.
- A shared acceptance boundary for preprocessing and the complete model path.
