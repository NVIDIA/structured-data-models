# StypeDispatch Design

`StypeDispatch` is a processing container that applies different processors to
different semantic column types in a `TableTensor`, then merges the processed
tables back into one `TableTensor`.

The main use case is a recipe that needs stype-specific preprocessing while
keeping the public processing contract table-in/table-out. Some processors in
this example are future design targets rather than current exports:

```python
from sdm.processing import (
    Choice,
    ConstantFilter,
    FeaturePermute,
    Identity,
    MeanImpute,
    Quantile,
    Recipe,
    SigmaClip,
    StandardScale,
    StypeDispatch,
)

recipe = Recipe(
    features=[
        StypeDispatch({"categorical": ToNumerical()}),
        MeanImpute(),
        ConstantFilter(),
        StandardScale(epsilon=1e-6),
        Choice([Identity(), Quantile(output_distribution="normal")]),
        SigmaClip(threshold=4.0),
        FeaturePermute(method="latin"),
    ],
    target=[
        StypeDispatch(
            {
                "categorical": ClassShuffle(method="shift"),
                "numerical": StandardScale(),
            }
        ),
    ],
)
```

This should work with any `Stype`, with list-style values for multi-step
routes, and with processors that change the number, names, or order of
columns.

## Related Open PRs

The design depends on the current TableTensor and processing stack:

- [#193](https://github.com/NVIDIA/structured-data-models/pull/193) adds
  `TableTensor.replace_blocks`, which is the table-native way for processors to
  return changed blocks without manually rebuilding unrelated stypes.
- [#194](https://github.com/NVIDIA/structured-data-models/pull/194) adds
  `TableTensor.select_stype`, which `StypeDispatch` should use to pass a
  stype-only `TableTensor` to each route.
- [#197](https://github.com/NVIDIA/structured-data-models/pull/197) changes
  processing pipelines to accept and return `TableTensor`s, which is the public
  contract `StypeDispatch` relies on.
- [#199](https://github.com/NVIDIA/structured-data-models/pull/199) adds
  processor `supported_stypes`, which lets routed processors reject non-empty
  unsupported stypes after dispatch.

The example processors are also covered by open draft PRs and should align with
this design if they continue:

- [#148](https://github.com/NVIDIA/structured-data-models/pull/148) explores
  table-level processor steps.
- [#149](https://github.com/NVIDIA/structured-data-models/pull/149) adds a
  feature permutation processor.
- [#150](https://github.com/NVIDIA/structured-data-models/pull/150) adds
  constant filtering and label shuffle processors.
- [#155](https://github.com/NVIDIA/structured-data-models/pull/155) adds a
  `ToNumerical` processor.

## AGENTS.md Alignment

This design follows the repository guidance in `AGENTS.md`:

- Keep preprocessing generic, composable, and tensor-centric.
- Preserve direct Python composition as the primary interface instead of adding
  mandatory config-first APIs.
- Keep model-facing processing table-in/table-out on `TableTensor`.
- Prefer existing local abstractions, especially `Processor`, `Sequential`,
  `Recipe`, `TableTensor.select_stype`, `TableTensor.replace_blocks`, and
  column-wise `torch.cat`.
- Allow list-style route values for user ergonomics, but keep normalization
  local to `StypeDispatch.__init__` for now. A shared helper can be added later
  if both `Recipe` and `StypeDispatch` need the exact same reusable logic.

## Goals

- Keep every public processor method table-in/table-out:
  `fit`, `transform`, `fit_transform`, and `forward` receive a
  `TableTensor` and return a `TableTensor`.
- Let processors declare the stypes they can handle and fail early when a
  non-empty unsupported stype reaches them.
- Route each configured stype through the processor assigned to that stype.
- Preserve unconfigured stypes with an explicit remainder policy.
- Support shape-changing processors, for example categorical encoders, SVD,
  constant filters, class shuffling, and feature permutation.
- Keep the implementation PyTorch-native and use `TableTensor` operations for
  slicing and column-wise concatenation.

## Proposed API

```python
StypeDispatch(
    processors: Mapping[StypeLike, Processor | Iterable[Processor]],
    *,
    remainder: Literal["passthrough", "drop", "error"] = "error",
)
```

`processors` maps a stype to the processor route that should receive only that
stype's columns. A value may be a single `Processor` or an iterable of
processors. Iterable values should be normalized to `Sequential`, matching the
role-level ergonomics of `Recipe`:

```python
StypeDispatch({
    "categorical": [ToNumerical(), MeanImpute()],
    "numerical": StandardScale(),
})
```

This keeps the user-facing API convenient without requiring users to spell
`Sequential(...)` inside every multi-step route. For the first implementation,
the normalization can stay inline in `StypeDispatch.__init__`; a private helper
is not necessary until the same logic is reused in enough places to remove real
duplication.

`remainder` controls non-empty stypes that are present in the input but not
configured:

- `"error"` raises. This is the safest default because unprocessed columns are
  never silently carried through a recipe.
- `"passthrough"` carries unconfigured stypes unchanged.
- `"drop"` removes unconfigured stypes.

`StypeDispatch` itself should set `supported_stypes = "all"` because it is a
container. Each routed processor remains responsible for validating the stype
slice it receives.

## Core Semantics

For each public method, `StypeDispatch` should:

1. Validate and normalize configured stypes with `Stype(stype)`.
2. Select the input table for each configured stype.
3. Call the corresponding processor method on that stype-only table.
4. Apply the configured remainder policy to unconfigured stypes.
5. Merge all resulting `TableTensor`s with `torch.cat(..., dim=-1)`.

The selected table for one stype must still be a `TableTensor`, not the raw
block tensor. For example, the categorical route receives a `TableTensor` with
only categorical columns. This keeps every processor API consistent and lets a
route convert categorical columns into numerical columns by returning a
different stype than it received.

The merge order should be deterministic:

1. outputs from configured routes in the user-provided mapping order;
2. remainder outputs in the original `TableTensor` stype order.

This preserves Python mapping ergonomics, makes column order inspectable from
the recipe definition, and avoids tying route output order to the internal
`TableTensor` storage order.

## Fit And Transform

`fit` and `fit_transform` need leakage-safe routed behavior. During `fit`, each
processor only sees the stype-specific slice it owns:

```python
for stype, processor in self.processors.items():
    processor.fit(input.select_stype(stype))
```

During `fit_transform`, each processor should use its own `fit_transform` so
stateful processors can learn on the same slice they transform and return the
transformed route output:

```python
outputs = []
for stype, processor in self.processors.items():
    outputs.append(processor.fit_transform(input.select_stype(stype)))
outputs.extend(self._remainder_outputs(input))
return torch.cat(outputs, dim=-1)
```

`transform` and `forward` should route through `processor.transform(...)` so
fitted-state checks stay inside the processor contract. If the transform API PR
removes or changes the role of `forward`, the dispatch implementation should
follow the final base `Processor` contract and keep only one internal routing
path.

## Required TableTensor Helpers

`StypeDispatch` should depend on the `TableTensor.select_stype` helper from
PR #194 and on column-wise `torch.cat` for recombining route outputs:

```python
table.select_stype(stype) -> TableTensor
torch.cat([table_a, table_b], dim=-1) -> TableTensor
```

`select_stype` returns a `TableTensor` with the requested stype populated and
unselected stypes represented as empty blocks. PR #194 implements this with a
direct block path that reuses the selected block without copying and relies on
the `TableTensor` constructor for schema validation. Its local benchmark found
that path faster than a `select_columns` wrapper for the measured cases, so
`StypeDispatch` should treat `select_stype` as the public dependency and not
reimplement stype slicing itself.

## Edge Cases

Empty stype slices should be skipped unless the configured processor explicitly
needs to see empty inputs. Skipping avoids fitting numerical processors on
zero-column tables and keeps no-op routes cheap.

Duplicate output column names should raise through the `TableTensor`
constructor during merge. This keeps the uniqueness rule centralized.

If a route changes row or batch dimensions, `torch.cat(..., dim=-1)` should
raise. A processor that changes rows is not a column transform and should not be
used inside `StypeDispatch`.

If all outputs are dropped or empty, the dispatch should return an empty
`TableTensor` with the original batch shape and device. This needs an explicit
helper because `torch.cat([])` is invalid.

## Implementation Sketch

```python
class StypeDispatch(Processor):
    supported_stypes = "all"

    def __init__(self, processors, *, remainder="error"):
        super().__init__()
        self.processors = ModuleDict()
        for stype, processor in processors.items():
            if not isinstance(processor, Processor):
                processor = Sequential(*processor)
            self.processors[Stype(stype).value] = processor
        self.remainder = remainder
        self.requires_fit = any(
            processor.requires_fit for processor in self.processors.values()
        )

    def fit(self, input):
        for stype, processor in self._processors_by_stype():
            route_input = input.select_stype(stype)
            if route_input.size(-1) == 0:
                continue
            processor.fit(route_input)
        self._fitted = True
        return self

    def fit_transform(self, input):
        outputs = []
        for stype, processor in self._processors_by_stype():
            route_input = input.select_stype(stype)
            if route_input.size(-1) == 0:
                continue
            outputs.append(processor.fit_transform(route_input))
        outputs.extend(self._remainder_outputs(input))
        self._fitted = True
        return self._merge_outputs(input, outputs)
```

`ModuleDict` is useful for PyTorch registration, but the public stype keys
should still be normalized back to `Stype` when routing. A private helper such
as `_processors_by_stype()` can hide the string-key conversion.

## Why This Design

The dispatch processor stays consistent with the rest of processing: it is a
`Processor`, it composes inside `Sequential`, and it keeps direct Python
composition as the main API. It avoids a special recipe-only code path and lets
future processors reuse the same route-and-merge mechanism.

The design is also robust for forward shape-changing processors because the
route output is a full `TableTensor`. A categorical encoder can return
numerical columns, a constant filter can remove columns, and a feature
permutation can change ordering without requiring special cases in
`Sequential` or `Recipe`. Inverse behavior is intentionally covered in a
separate design note.
