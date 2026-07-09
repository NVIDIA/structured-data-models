# Recipe Target Output Wrapping Design

This note isolates the model-output boundary problem from
`StypeDispatch.inverse_transform`. The design is intentionally deferred:
`StypeDispatch` should first support inverse for an already-structured
`TableTensor`.

## Problem

Current model APIs return raw `torch.Tensor` predictions, while target
processors consume and return `TableTensor`. A raw tensor contains values and
shape but not transformed target names, stypes, or route ownership.

That missing metadata cannot be reconstructed by `StypeDispatch`. Dispatch
only owns route processing after a valid transformed table exists.

## Ownership

`Recipe` is the appropriate future owner because it defines the external model
boundary:

- `recipe.features` prepares model inputs;
- `recipe.target` prepares labels and inverts predictions;
- `recipe.output` applies post-inverse output processing.

A future adapter should therefore sit on `Recipe`, not on
`StypeDispatch` or every processor.

## Option 2: Add A Recipe Target Output Wrapper

An illustrative API is:

```python
model_target = recipe.fit_transform_target(target)
raw_prediction = model(...)
prediction_table = recipe.wrap_target_output(raw_prediction)
prediction = recipe.target.inverse_transform(prediction_table)
```

`fit_transform_target` would run `recipe.target.fit_transform(target)` and
record the resulting transformed target schema. `wrap_target_output` would use
that fitted schema to construct the `TableTensor` expected by target inverse.

These names are design placeholders, not proposed implementation in this PR.

## Fitted Target Schema

The Recipe-owned state needs enough metadata to validate and reconstruct the
transformed target:

```python
@dataclass(frozen=True)
class TargetOutputSchema:
    columns: Mapping[Stype, tuple[str, ...]]
    width: int
```

Additional block offsets and dtypes are necessary if transformed targets can
contain multiple stypes. A homogeneous raw tensor cannot reconstruct arbitrary
typed blocks without an explicit layout convention.

## Validation

A future wrapper should:

- require target fitting before wrapping output;
- require the prediction's last dimension to equal transformed target width;
- allow different leading batch dimensions from the fitted target;
- preserve prediction device and dtype where compatible;
- restore recorded names and per-stype order;
- reject schemas that cannot be represented from one homogeneous tensor.

It should not validate against the training batch size because inference batch
shape can differ.

## Alternative: Require TableTensor Model Output

Models could instead return a `TableTensor` matching the transformed target
schema. Recipe would still own validation at the model boundary, but no raw
tensor wrapper would be required.

Changing every model output type is a larger API decision. It is not required
for `StypeDispatch` inverse and is intentionally left open here.

## Decision For Now

Defer Recipe target-output wrapping.

The immediate `StypeDispatch` inverse contract requires a correctly structured
`TableTensor`. This allows route-owned inverse design to proceed independently
from model output API decisions.

## Open Questions

- Should Recipe expose `fit_transform_target` and `wrap_target_output`, or should
  a dedicated fitted target adapter own both operations?
- Can raw wrapping initially support numerical-only transformed targets?
- How should a raw tensor encode multiple stype blocks with different dtypes?
- Should models eventually return `TableTensor` directly?
- Should Recipe combine wrapping and inverse into one method, such as
  `inverse_target_output(raw_prediction)`?
