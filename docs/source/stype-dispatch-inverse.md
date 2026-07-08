# StypeDispatch Inverse Design

This note defines the inverse routing implemented by `StypeDispatch`. It
assumes the caller provides a `TableTensor` in transformed target space.
Adapting raw model tensors remains a separate Recipe boundary concern,
described in
[Recipe Target Output Wrapping Design](recipe-target-output.md).

## Current Contract

Forward transformation records each route's output width per semantic type.
Inverse transformation then:

- walks routes in the same order used by forward concatenation;
- slices each stype block into route-owned segments using the recorded widths;
- passes each segment to that route's `inverse_transform`;
- appends unconsumed columns as the passthrough remainder.

This supports routes that change stype or width. For example, a categorical
route may emit several numerical one-hot columns while another route also emits
numerical columns.

Column names are not stored or compared. The incoming prediction may use
different names from the transformed training target.

Every non-empty route must implement `inverse_transform` and accept its
transformed stypes under the current shared `supported_stypes` contract.

## Inverse Flow

The core algorithm maintains one offset per output stype:

```python
offsets = {stype: 0 for stype in Stype}
outputs = []

for route, processor in configured_routes:
    widths = recorded_output_widths[route]
    route_columns = []

    for stype, width in widths.items():
        start = offsets[stype]
        route_columns.extend(input.columns[stype][start : start + width])
        offsets[stype] += width

    route_output = input.select_columns(route_columns)
    outputs.append(processor.inverse_transform(route_output))

outputs.append(select_unconsumed_remainder(input, offsets))
return torch.cat(outputs, dim=-1)
```

`TableTensor` concatenation preserves route segment order within every stype
block, which makes width-based partitioning sufficient while that order is
retained.

If no forward layout has been recorded, a route falls back to selecting its
configured input stype. This preserves the previous behavior for
schema-preserving routes.

## TabICL Use

TabICLv2 regression transforms one numerical target with `StandardScale`. Its
inverse is positional and has one route segment, so prediction column names do
not need to equal the fitted target name.

The width-based layout also covers a future classification route that expands
one categorical target into several numerical columns. Constructing the
prediction `TableTensor` from raw model output remains a Recipe concern.

## When Ordering Fails

Ordering fails only when the prediction layout differs from the forward layout
whose widths were recorded.

For example, suppose forward concatenation produces one numerical block:

```text
[numerical route: x0, x1] [one-hot route: kind_a, kind_b]
```

The recorded widths are two and two. If model output swaps the route segments:

```text
[kind_a, kind_b] [x0, x1]
```

dispatch still sends the first two values to the numerical inverse and the last
two to the one-hot inverse. Both routes receive the wrong values.

Ordering can also fail inside one segment. Swapping `kind_a` and `kind_b`
changes which category an argmax-based one-hot inverse returns.

Names do not cause the failure; changed positions do. The current single-column
TabICL regression target has no meaningful ordering ambiguity. Recipes with
multiple outputs must preserve the forward segment and within-segment order.

## Remainder Behavior

- `remainder="error"` rejects unconsumed non-empty blocks.
- `remainder="passthrough"` returns unconsumed columns unchanged.
- `remainder="drop"` is never invertible.

## Future Extension

If reordered predictions need to be accepted or rejected explicitly, dispatch
can extend the recorded widths to full route schemas:

```python
@dataclass(frozen=True)
class RouteSchema:
    route_stype: Stype
    output_columns: Mapping[Stype, tuple[str, ...]]
```

Recorded names would allow dispatch either to validate exact order or to
reorder each route segment before inversion. The current width-based state is a
small subset of that design and keeps the public API unchanged.

## Out Of Scope

`StypeDispatch` does not construct a `TableTensor` from raw model output and
does not depend on TabICL or another model family. Recipe-level target output
adaptation remains deferred in the separate design note.
