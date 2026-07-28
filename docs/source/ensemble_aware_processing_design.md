# Ensemble-Aware Processing: Implementation Design

This document specifies the recommended solution to
[Ensemble-Aware Processing: Problem and Goals](ensemble_aware_processing_problem.md).

## Decisions

- `Recipe` is the only public ensemble-aware processing entry point. It receives
  task features, target, and related tables together.
- Every logical table has one fit scope and one Processor tree for all members.
  Version 1 does not share fitted state across tables.
- Processors own their ensemble semantics. `Recipe` does not contain special
  cases for `Choice`, `Sequential`, or `StypeDispatch`.
- Every Processor logically operates on all members `[E, ...]`.
  `EnsembleTable` may physically store only the distinct, stack-compatible
  variants `[V, ...]`.
- The base Processor supplies a correct per-member fallback. Sharing and direct
  vectorization require explicit opt-in.
- The public model API and existing member-wise `_forward` execution remain
  unchanged in version 1.

## Public API and Ownership

`Recipe` normalizes each of `features`, `target`, and `output` to one
`Sequential`. There is no second public ensemble-processing path.

```python
x_context, y_context, related_context = recipe.fit_transform(
    features=x_context,
    target=y_context,
    related_tables=related_context_tables,
    num_members=num_estimators,
    generator=generator,
)
x_query, related_query = recipe.transform(
    features=x_query,
    related_tables=related_query_tables,
)
prediction = recipe.transform_output(raw_member_outputs)
```

The return contracts are:

- `fit_transform`:
  `tuple[EnsembleTable, EnsembleTable, EnsembleRelatedTables | None]`
- `transform`: `tuple[EnsembleTable, EnsembleRelatedTables | None]`
- The related-table value is `None` when no related tables were provided.

The model creates one working copy of the configured Recipe with
`copy.deepcopy`, leaving the caller's configuration unchanged. Before fitting,
Recipe copies its unfitted feature pipeline once for every related table. The
task table, target, and every related table therefore each own one Processor
tree for all members. Fitted state is scoped to the concrete Processor node and
logical table. Recipe and model caches are installed only after a completely
successful fit.

A simplified flow is:

```python
def fit_transform(
    self,
    *,
    features: TableTensor,
    target: TableTensor,
    related_tables: RelatedTables | None,
    num_members: int,
    generator: torch.Generator | None = None,
) -> tuple[EnsembleTable, EnsembleTable, EnsembleRelatedTables | None]:
    member_ids = tuple(range(num_members))
    x = EnsembleTable.broadcast(features, member_ids=member_ids)
    y = EnsembleTable.broadcast(target, member_ids=member_ids)

    related_processors = (
        {
            name: copy.deepcopy(self.features)
            for name in related_tables.tables
        }
        if related_tables is not None
        else {}
    )

    x = self.features._fit_transform_ensemble(x, generator=generator)
    y = self.target._fit_transform_ensemble(y, generator=generator)

    related_outputs = {}
    if related_tables is not None:
        for name, table in related_tables.tables.items():
            inp = EnsembleTable.broadcast(table, member_ids=member_ids)
            related_outputs[name] = related_processors[
                name
            ]._fit_transform_ensemble(inp, generator=generator)

    self._resolve_output_task(y)
    self._related_features = related_processors
    self._num_members = num_members
    return x, y, _ensemble_related(related_tables, related_outputs)
```

This example shows ownership and data flow. The exact order of RNG-consuming
operations remains an open decision.

## Ensemble Containers

```python
@dataclass(frozen=True, repr=False)
class EnsembleTable:
    # Stable logical IDs. Subsets retain the original IDs.
    member_ids: tuple[int, ...]
    # Group i contains stack-compatible variants [V_i, ..., R, C_i].
    variant_groups: tuple[TableTensor, ...]
    # Position in member_ids -> (group index, variant index).
    member_to_variant: tuple[tuple[int, int], ...]

    @property
    def num_members(self) -> int: ...

    def __getitem__(self, member_id: int) -> TableTensor: ...

    def select_members(
        self,
        member_ids: Sequence[int],
    ) -> EnsembleTable: ...

    @classmethod
    def broadcast(
        cls,
        table: TableTensor,
        *,
        member_ids: Sequence[int],
    ) -> EnsembleTable: ...

    @classmethod
    def merge_members(
        cls,
        parts: Sequence[EnsembleTable],
        *,
        member_ids: Sequence[int],
    ) -> EnsembleTable: ...
```

The invariants are:

- The container semantically represents member-ordered input `[E, ..., R, C]`;
  this shape need not be physically materialized.
- `member_ids` remain stable through `select_members`, preserving nested
  `Choice`, round-robin, and RNG semantics.
- A variant group has one shape, column schema and order, stypes, stype-local
  dtypes, categorical metadata, and device.
- Schema or column-count changes create separate groups.
- Multiple members may reference the same physical variant.
- The initial input contains one variant `[1, ..., R, C]` referenced by all
  members.
- `table[e]` resolves stable member ID `e` and returns a `TableTensor` view where
  possible. Version 1 supports integer member IDs only.
- `EnsembleTable` is not a `TableTensor` subclass. `batch` remains reserved for
  real tensor batches or chunking.

Related tables retain their graph metadata:

```python
@dataclass(frozen=True, repr=False)
class EnsembleRelatedTables:
    tables: Mapping[str, EnsembleTable]
    relationships: tuple[Relationship, ...]
    task_links: tuple[TaskLink, ...]

    @property
    def num_members(self) -> int: ...

    def __getitem__(self, member: int) -> RelatedTables:
        return RelatedTables(
            tables={
                name: table[member]
                for name, table in self.tables.items()
            },
            relationships=self.relationships,
            task_links=self.task_links,
        )
```

All tables have the same logical member count. `related_tables[e]` returns
normal `TableTensor` views, so the model API does not need to change.

Version 1 executes an ensemble entirely on one device. The container does not
encode transfer or placement semantics; `.to()`, `.device`, and multi-device
placement require a separate design.

## Processor Contract

Standalone Processors retain their public `TableTensor` API. Internally, every
Processor supports ensemble hooks; the normal case is one member with one
variant:

```python
class Processor:
    def _fit_transform_ensemble(
        self,
        table: EnsembleTable,
        *,
        generator: torch.Generator | None = None,
    ) -> EnsembleTable: ...

    def _transform_ensemble(
        self,
        table: EnsembleTable,
    ) -> EnsembleTable: ...


class InvertibleMixin:
    def _inverse_transform_ensemble(
        self,
        table: EnsembleTable,
    ) -> EnsembleTable: ...
```

`_fit_transform_ensemble` registers fitted state on the invoked Processor
instance. Transform and inverse operations use only this state and align it by
logical member ID, not by physical group index. This is required for query reuse
and target inversion because those inputs may be grouped differently from the
fit input.

Every hook processes $E$ logical members and returns exactly one associated
result per member:

- The base class executes unknown Processors independently per member.
- `Choice` records one option per member, groups equal options, and executes
  their branches.
- Directly vectorizable leaves process distinct compatible variants as
  `[V, ..., R, C]`.
- Compatible branch outputs may stay stacked. Different schemas or metadata
  create separate variant groups.
- `EnsembleReduce` is the only operation that reduces the logical member
  dimension.

`[E, ...]` is the required semantic contract; `[V, ...]` is an optional physical
optimization. Requiring a dense `[E, ...]` tensor would materialize and recompute
identical members and could not represent member-dependent column counts.

| Contract             | Semantics                                                                            | Current Processors                                                                                                                                                                                               |
| -------------------- | ------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| base `Processor`     | Correct independent per-member fallback; no direct vectorization                     | `Callable`, unknown and external Processors                                                                                                                                                                      |
| deterministic fit    | No RNG use; fit and execute every distinct input variant once                        | `CategoricalAlign`, `ConstantFilter`                                                                                                                                                                             |
| vectorized variants  | Deterministic fit plus independent execution of compatible `[V, ..., R, C]` variants | Direct: `Identity`, `Clip`, `ToNumerical`, `EncodeDatetime`, `SoftmaxTemperature`; small axis changes: `MeanImpute`, `StandardScale`, `QuantileClip`, `SigmaClip`; full adaptation: `Power`, `CategoricalImpute` |
| custom ensemble hook | Own member-specific RNG, split, merge, or metadata semantics                         | `Choice`, `FeaturePermute`, `CategoryShuffle`, `Quantile`                                                                                                                                                        |
| composite Processor  | Delegate `EnsembleTable` to children and combine outputs                             | `Sequential`, `StypeDispatch`, `TaskDispatch`                                                                                                                                                                    |
| aggregation boundary | Explicitly reduce the member dimension                                               | `EnsembleReduce`                                                                                                                                                                                                 |

`Quantile` needs a custom hook for RNG-based subsampling. It may use the
deterministic path when subsampling is inactive.

The stronger contracts are required for safe `[V, ...]` vectorization. A
`supports_stacked_variants` flag cannot guarantee independent fitted state,
safe reuse, or compatible output metadata. No generic column-changing or
ensemble-stype mixin is needed:

- `supported_stypes` continues to validate leaf inputs.
- `StypeDispatch` owns route split and recombination.
- Column changes may be schema-fixed (`ToNumerical`), fit-dependent
  (`ConstantFilter`), or member-specific (`FeaturePermute`).
- Outputs are regrouped from their complete resulting metadata.

Equivalent members do not create duplicate state. For example, `MeanImpute`
may process a compatible `[V, R, C]` group with one fitted instance and store
means as `[V, 1, C]`. Incompatible groups require separate fitted states; their
representation is an open implementation question.

## Execution

### Deterministic Leaves

In the mean/variance case, `StandardScale` needs no custom ensemble hook. A
vectorized-variant implementation passes each compatible physical group
`[V, ..., R, C]` through the normal implementation and registers group-specific
state:

```python
class StandardScale(VectorizedVariantsMixin, Processor, InvertibleMixin):
    def _fit(self, table: TableTensor, *, generator=None) -> None:
        values = table.numerical
        self.mean = values.mean(dim=-2, keepdim=True)
        var = values.var(dim=-2, correction=0, keepdim=True)
        self.scale = _scale_from_variance(
            var,
            self.mean,
            num_rows=values.size(-2),
        )

    def _transform(self, table: TableTensor) -> TableTensor:
        return table.replace_blocks(
            numerical=(table.numerical - self.mean) / self.scale,
        )

    def _inverse_transform(self, table: TableTensor) -> TableTensor:
        return table.replace_blocks(
            numerical=table.numerical * self.scale + self.mean,
        )
```

The state shape is `[1, C]` for a normal input and `[V, ..., 1, C]` for
$V$ variants.

`ConstantFilter` instead uses the deterministic-fit contract. It executes
different variants separately because their output schemas may differ:

```python
class DeterministicFitMixin:
    def _fit_transform_ensemble(self, table, *, generator=None):
        fitted_groups = []
        parts = []
        for variant in table.unique_variants():
            fitted = self._new_unfitted_group_instance()
            out = fitted.fit_transform(
                table[variant.representative_member_id]
            )
            part = EnsembleTable.broadcast(
                out,
                member_ids=variant.member_ids,
            )
            fitted_groups.append((variant.member_ids, fitted))
            parts.append(part)

        self._register_fitted_groups(fitted_groups)
        return EnsembleTable.merge_members(
            parts,
            member_ids=table.member_ids,
        )
```

A future specialization may compute masks jointly for `[V, C]`, but outputs
with different masks still form separate groups.

### Composite and Stochastic Processors

`Sequential` passes `EnsembleTable` through its steps without Processor-specific
knowledge:

```python
class Sequential(Processor, InvertibleMixin):
    def _fit_transform_ensemble(self, table, *, generator=None):
        out = table
        for step in self:
            out = step._fit_transform_ensemble(
                out,
                generator=generator,
            )
        return out

    def _transform_ensemble(self, table):
        out = table
        for step in self:
            out = step._transform_ensemble(out)
        return out

    def _inverse_transform_ensemble(self, table):
        out = table
        for step in reversed(self):
            out = step._inverse_transform_ensemble(out)
        return out
```

`Choice` implements the `[E, ...]` contract itself:

```python
class Choice(Processor, InvertibleMixin):
    def _fit_transform_ensemble(self, table, *, generator=None):
        selected = self._select_in_reference_order(
            member_ids=table.member_ids,
            generator=generator,
        )
        self._selected_by_member = dict(
            zip(table.member_ids, selected)
        )

        parts = []
        for option_index in dict.fromkeys(selected):
            member_ids = tuple(
                member_id
                for member_id in table.member_ids
                if self._selected_by_member[member_id] == option_index
            )
            parts.append(
                self.options[
                    option_index
                ]._fit_transform_ensemble(
                    table.select_members(member_ids),
                    generator=generator,
                )
            )
        return EnsembleTable.merge_members(
            parts,
            member_ids=table.member_ids,
        )
```

Round-robin uses the stable original `member_id % num_options`. Compatible
branch outputs are stacked; incompatible outputs remain in separate groups.

`StypeDispatch` selects active stype columns for every variant group, delegates
each route to its child Processor, retains passthrough columns according to
`remainder`, and regroups the reconstructed member outputs.

`FeaturePermute` creates one permutation per member and applies compatible
permutations jointly with `gather`. Because one `TableTensor` has one column
order across all leading dimensions, version 1 creates one group per distinct
order. `CategoryShuffle` follows the same principle for category mappings.
Query transform reuses stored mappings; inverse transform uses their inverses.

### RFM

- Task table, target, and every related table are separate fit scopes.
- Every related table owns one feature-Processor copy for all members.
- Deterministic prefixes and text or categorical encoders are shared within a
  table.
- Feature permutations remain table-local.
- One estimator-owned RNG plan may derive stable substreams for table-local
  stochastic work.
- Ownership of feature-level `Choice` across tables remains unresolved. The
  implementation must not encode repeated same-seed resets as the contract.
- `relationships` and `task_links` remain shared graph metadata.
- Cross-table fitted-state sharing is outside version 1.

## Model, Target, and Output Integration

Processing moves before the existing member loop:

```python
recipe = copy.deepcopy(recipe)
x_context, y_context, related_context = recipe.fit_transform(...)
x_query, related_query = recipe.transform(...)

outs = [
    self._forward(
        x_context=x_context[e],
        y_context=y_context[e],
        x_query=x_query[e],
        related_context_tables=(
            related_context[e]
            if related_context is not None
            else None
        ),
        related_query_tables=(
            related_query[e]
            if related_query is not None
            else None
        ),
        cache=member_caches[e] if member_caches else None,
        generator=generator,
        **kwargs,
    )
    for e in range(num_estimators)
]
return recipe.transform_output(outs)
```

- `_forward`, `ICLModel.forward`, `fit`, and `predict` remain publicly
  unchanged.
- `fit` stores one fitted Recipe and one model/KV cache per member.
- `predict` transforms the complete query graph once, then selects member views.
- Regression applies the member-aligned target
  `_inverse_transform_ensemble`.
- Classification aligns member-specific logit columns after `CategoryShuffle`
  into one class space.
- Outputs are then materialized exactly once in member order as
  `[E, ..., R, O]` and passed through `recipe.output.transform`.
- Only `EnsembleReduce` may aggregate members.

## Implementation Order

1. Add `EnsembleTable`, `EnsembleRelatedTables`, internal hooks, and the correct
   per-member fallback.
2. Add Recipe, `Sequential`, simple deterministic leaves, `Choice`, `Power`,
   `FeaturePermute`, and the target/output path needed by TabICLv2.
3. Demonstrate strict TabICLv2 parity and benchmark the target workload.
4. Add remaining built-in Processors, `StypeDispatch`, and RFM related tables.
5. Specify cross-table scheduling, multi-device placement, or models consuming
   grouped inputs directly only after need is demonstrated.

The smallest maintainable end-to-end slice is therefore the generic container
and Processor contract plus the hooks needed for TabICLv2, not a
`Power`-specific cache.

## Validation

Parity coverage must include:

- shared deterministic prefixes and suffixes;
- random and round-robin `Choice`;
- `StypeDispatch`, column-count changes, and member-specific permutations;
- query-state reuse, regression inverse transform, and class alignment;
- `EnsembleReduce`;
- RFM with multiple tables;
- unknown Processor fallback;
- negative contract tests: no sharing without opt-in, no RNG use by the
  deterministic-fit contract, and equality of vectorized variants with
  independent execution within the agreed tolerance.

The GPU benchmark must:

- compare current main, sharing without vectorization, direct variant execution,
  and the fastest valid baseline;
- use 40k context rows, 10k query rows, 100 features, and eight members split
  into four `Power` and four `Identity` variants;
- measure the complete target path, including processing, member
  materialization, model call, and output processing;
- use GPU-resident inputs, warm-up, CUDA events, and explicit synchronization;
- report median, p95, kernel time, transfers, synchronization, peak memory, and
  throughput;
- remeasure both the previous 0.46–0.47 second result and the 0.19–0.20 second
  target using the same boundary.

## Hard Boundaries

- Processors preserve row count; context and query may have different row
  counts.
- Randomness occurs only during `fit`; `transform` does not mutate state.
- No state is reused across tables, devices, or non-equivalent fit scopes.
- No implicit CPU fallback, transfer, or content hashing.
- Variants may be preserved or split; only `EnsembleReduce` aggregates members.
- Version 1 excludes row-count changes, multi-GPU execution, and cross-table
  fitted-state sharing.

## Open Implementation Questions

- **I1 — Fitted groups:** Should incompatible states be represented as normal
  child Processors or in a generic serializable state container?
- **I2 — RNG plan:** How is reference draw order preserved for nested and
  data-dependent randomness? For RFM, is feature-level `Choice` owned by the
  estimator or by each table fit scope?
- **I3 — Group operations:** Which private select, merge, regroup, and
  member-state alignment helpers are needed by composite Processors, query
  transform, and target inverse transform?
- **I4 — State lifecycle:** How do ensemble-aware states behave on refit,
  deep-copy, serialization, freeze, and a future `.to()`?
- **I5 — Vectorization threshold:** At what variant or table size does joint
  execution outperform separate execution?

## Alternative: Recipe-Owned Fitted Nodes

A Recipe-owned `_FittedNode` could return a separate state tree containing
child nodes and member mappings alongside each output. This makes state
ownership explicit but introduces a second execution representation parallel
to the Processor tree. The recommended Processor-owned design keeps fit,
transform, inverse transform, deep-copy, and serialization semantics in the
existing abstraction. Reconsider `_FittedNode` only if Processor-owned grouped
state proves unmanageable during implementation.
