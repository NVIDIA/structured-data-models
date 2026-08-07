:orphan:

# Processor `operates_on_stypes` Spec

## User View

Recipes should make passthrough explicit at the pipeline boundary, not on each processor. A user can write a feature recipe that transforms numerical columns while preserving identifiers:

```python
recipe = Recipe(features=[Standardize()], feature_passthrough_stypes={Stype.id})
```

`Standardize` only declares that it operates on numerical columns. The recipe declares that feature identifiers may pass through unchanged. Unexpected feature stypes still fail early at the recipe/model boundary:

- `numerical + id`: allowed, numerical is transformed, id is preserved.
- `numerical + text`: rejected unless the recipe explicitly allows text passthrough or routes it.
- `id` only: allowed as passthrough when the recipe allows id.

## Naming

Use `operates_on_stypes`, plural, replacing `supported_stypes`. The plural matches the current `supported_stypes` API and the value type (`frozenset[Stype]`). “Operates on” is more precise than “supports”: it describes which blocks a processor reads, fits, transforms, converts, or drops. It does not mean every other block is invalid.

## Implementation Details

`Processor` should use `operates_on_stypes` only for lifecycle gating: if no operated stype is active, `fit`, `transform`, and `fit_transform` are no-ops and do not require fitted state. The current `_check_supported_stypes` validation should move out of leaf processors and into recipe/model boundaries, where passthrough policy is known.

For fixed processors, `operates_on_stypes` can stay a class attribute. For configurable containers, expose it dynamically:

- `Sequential`: union of child `operates_on_stypes`; empty sequence operates on no stypes.
- `Choice`: union of option `operates_on_stypes`.
- `StypeDispatch`: configured route stypes for `remainder="passthrough"`/`"error"`; all stypes for `remainder="drop"` because unconfigured blocks are intentionally consumed by dropping.
- `Callable`: all stypes, because the callable contract is opaque.

Recipe validation should compute allowed input stypes as `features.operates_on_stypes | feature_passthrough_stypes`. PR350 currently hard-codes feature id passthrough by normalizing `Recipe.features` into `Sequential(..., passthrough_stypes={Stype.id})`; this can become an explicit recipe boundary option while preserving the default `{Stype.id}` if that remains the desired user behavior.

## Processor Audit

Most processors already preserve foreign blocks because they call `table.replace_blocks(...)`: numerical clipping/imputation/standardization/power/quantile transforms, categorical align/impute/shuffle, softmax, identity, and wrapper processors. `PCA` and `AddCalendarFields` also preserve foreign blocks by concatenating `table.drop_stypes(...)` or the original table with their output.

Processors that would drop foreign blocks if base validation is relaxed:

- `ToNumerical`: returns a new numerical-only table. Easy fix: concatenate `table.drop_stypes({Stype.numerical, Stype.categorical})` with the converted numerical output.
- `TfidfTextEmbed`: returns a new numerical-only table. Easy fix: concatenate `table.drop_stypes(Stype.text)` with the generated numerical output.
- `ShuffleColumns`: rebuilds only the numerical block. Easy fix: return `torch.cat([table.drop_stypes(Stype.numerical), permuted_numerical_table], dim=-1)`.
- `DropConstantColumns`: selects only kept numerical columns. Easy fix: select kept numerical columns plus every non-numerical input column.
- `ReduceEstimators`: output-only reducer rebuilds numerical output after reducing an ensemble dimension. Keep strict boundary validation for output processors unless passthrough reduction semantics are explicitly defined.

`SelectColumns`, `StypeDispatch(remainder="drop")`, and user `Callable` processors may intentionally drop columns; tests should treat them as explicit column-selection or opaque behavior, not automatic preservation.

## Tests

Add processor contract tests that bypass recipe policy and assert foreign-block preservation for every built-in processor whose operation is not explicit selection/drop. Add recipe boundary tests for early rejection of unexpected stypes and allowed id passthrough. Add container tests for dynamic `operates_on_stypes` after `append`/`extend`, and for `StypeDispatch` route/drop/error semantics.
