:orphan:

# Processor Stype Semantics Spec

## User View

Users compose processors by operation: standardize numbers, embed text, route by stype, select columns, drop stypes, or reduce model outputs. A processor should only fail because the data is invalid for the operation it performs, not because the table carries extra blocks for downstream consumers.

Example: `Standardize()` operates on numerical columns. If a feature table also contains identifiers, numerical columns are standardized and identifiers remain available afterwards. If the final feature table still contains a stype that the model cannot consume, the model boundary fails or warns there.

The rule is: each processor declares which stypes it works on and what it promises to do with the other stypes.

## Semantics

The canonical processor field is:

```python
operates_on_stypes: ClassVar[frozenset[Stype]]
```

It means: these are the stypes the processor reads, fits, transforms, converts, selects, drops, or reduces.

Processors also declare their contract for active stypes outside `operates_on_stypes`:

```python
unoperated_stype_policy: ClassVar[
    Literal["preserve", "error", "opaque"]
] = "preserve"
```

- `"preserve"`: the processor accepts non-operated blocks and returns them unchanged. This is the default.
- `"error"`: non-operated blocks are invalid input. Use this when passthrough semantics are undefined, such as output reduction.
- `"opaque"`: behavior is delegated to user code or dynamic child selection and cannot be summarized statically.

Dropping is an operation, not a policy. `DropStypes(Stype.text)` operates on `text`, returns no text columns, and preserves every other stype.

## Ownership

`Processor` subclasses own the efficient implementation of their operation and any promised preservation. The base class owns only cheap, uniform lifecycle decisions:

- If no operated stype is active, `"preserve"` returns the input unchanged, `"error"` raises when active non-operated blocks exist, and `"opaque"` runs the implementation.
- If operated stypes are active, `"error"` first rejects active non-operated blocks; otherwise the base class calls the implementation with the original table.
- The base class does not split and recombine tables for `"preserve"`; that would add allocations and column concatenation. Preserve processors must keep foreign blocks themselves, usually with `table.replace_blocks(...)` or by concatenating only when their operation changes width or stype.

This keeps performance predictable: ordinary block transforms reuse the original table structure, and schema-changing processors pay only for their own operation.

Container metadata is derived from children: `Sequential` uses the union of child `operates_on_stypes` and escalates policy to `"opaque"` if any child is opaque, otherwise to `"error"` if any child errors, otherwise `"preserve"`. `Choice` is `"opaque"` because the fitted option decides behavior. `Callable` is also `"opaque"`.

`StypeDispatch` should be only a router: configured route stypes are operated; unconfigured stypes are either preserved or rejected by `remainder="error"`.

## Processor Audit

Most processors already fit `"preserve"` efficiently because they call `table.replace_blocks(...)`: numerical clipping/imputation/standardization/power/quantile transforms, categorical align/impute/shuffle, softmax, identity, and wrappers. `PCA` and `AddCalendarFields` already preserve by concatenating non-operated blocks with their output.

Processors needing small updates before they can honestly declare `"preserve"`:

- `ToNumerical`: concatenate non-numerical/non-categorical blocks with the converted numerical output.
- `TfidfTextEmbed`: concatenate non-text blocks with the generated numerical output.
- `ShuffleColumns`: use `replace_blocks` or concatenate non-numerical blocks with the permuted numerical output.
- `DropConstantColumns`: keep selected numerical columns plus every non-numerical input column.
- `DropStypes`: new explicit schema-changing processor; it operates on the dropped stypes and preserves the rest.

`ReduceEstimators` uses `"error"`: it reduces an ensemble dimension on numerical model outputs, and reduction semantics for ids, text, or categorical blocks are not defined.

## Tests

Add contract tests for preservation on every built-in `"preserve"` processor, including width- and stype-changing processors. Add `"error"` tests for `ReduceEstimators`. Add container tests for dynamic `operates_on_stypes` and policy aggregation. Add model-boundary tests for fail/warn behavior when final feature stypes are unsupported. PR351, `rbendias/tabiclv2-recipe-id-passthrough`, must continue to pass and should be rebased or merged onto the current `rbendias/recipe-feature-id-passthrough` state.

## Follow-up

Create a separate PR that removes drop semantics from `StypeDispatch` and adds explicit `DropStypes`. Then schema removal is expressed as its own step, for example `Sequential(DropStypes(Stype.text), StypeDispatch(numerical=Standardize()))`.
