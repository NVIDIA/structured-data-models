# Model-output inverse processing design

## Status

This document proposes the contract for applying target processors in reverse
to model outputs. It is a design, not an implementation.

The proposal complements the width-aware `StypeDispatch` inverse in
[PR #202](https://github.com/NVIDIA/structured-data-models/pull/202). That PR
partitions an already structured `TableTensor` using widths recorded during
forward transformation. Model outputs need additional information because one
transformed target column can produce a head with a different width, such as
`C` class scores or `Q` regression quantiles.

## Problem

The current processing API describes `inverse_transform` as a
`TableTensor -> TableTensor` mathematical inverse. Models currently return raw
`torch.Tensor` values, and their output dimensions do not necessarily
correspond one-to-one with transformed target columns:

- one categorical target is represented by one integer during training but by
  `C` logits during prediction;
- one numerical target is represented by one scalar during training but
  TabICLv2 returns `Q = 999` quantile values;
- a multi-target model may return separate heads with different widths and
  meanings.

A raw tensor contains shapes and values, but not target ownership or the
meaning of an axis. Forward output width, dtype, and column names cannot
reconstruct that information reliably.

`ClassShuffle` exposes the distinction most clearly. Inverting a transformed
class index uses the inverse permutation. Inverting class scores uses the
forward permutation to reorder the class axis. A processor cannot choose the
correct operation from a raw tensor alone.

## General rule

Let a fitted target processor define a transform

```text
f: original target space Y -> model target space Z.
```

`inverse_transform` maps a prediction expressed in `Z` coordinates back to
`Y` coordinates. The exact induced operation depends on the prediction
representation:

| Output representation                        | Induced inverse                             |
| -------------------------------------------- | ------------------------------------------- |
| Point value in `Z`                           | Apply `f^-1`                                |
| Samples, quantiles, or support values in `Z` | Apply `f^-1` to every value                 |
| Scores or probabilities over discrete `Z`    | Reindex the event axis into `Y` order       |
| Opaque distribution parameters               | Require an explicit processor-specific rule |

Processors only change axes and values owned by their fitted target columns.
They preserve batch, row, estimator, and unrelated head dimensions.

This is a pullback of a model prediction through the fitted target transform,
not necessarily a call to the same tensor operation used for label values.

## Typed model outputs

Model-output semantics must be explicit. The illustrative types below are not
final names or an implementation prescription:

```python
class OutputKind(Enum):
    POINT = auto()
    CLASS_SCORES = auto()
    QUANTILES = auto()
    SAMPLES = auto()


@dataclass(frozen=True)
class TargetRef:
    index: int
    name: str
    stype: Stype


@dataclass(frozen=True)
class OutputHead:
    values: Tensor
    target: TargetRef
    kind: OutputKind
    event_axis: int | None = None


@dataclass(frozen=True)
class ModelOutput:
    heads: tuple[OutputHead, ...]
```

The model, as producer of the output, owns `OutputKind`, head boundaries, and
axis semantics. The target pipeline owns the fitted original and transformed
target schemas and the processor route for each target. Neither side should
infer these facts from generated column names.

The initial contract assigns exactly one fitted target to each head. This
avoids an implicit target axis and supports targets with different
cardinalities. `TargetRef.index` is the stable identity; the name is diagnostic
metadata. A future coupled multi-target head needs an explicit target-axis
layout. A flattened representation can be an adapter that carries declared
slices into the raw tensor.

### Raw tensor compatibility

A raw `torch.Tensor` may be accepted only through a model-supplied output
specification. A single-target recipe can provide a convenient adapter because
the entire tensor belongs to one known head. Ambiguous multi-target raw tensors
must be rejected rather than partitioned heuristically.

## Recipe ownership and public API

`Recipe.target` should become a specialized target pipeline instead of a plain
`Sequential`. It owns:

- the target processors;
- original and transformed fitted target schemas;
- target-to-processor-route ancestry;
- the model-output specification or adapter.

The intended user flow remains:

```python
model_target = recipe.target.fit_transform(target)
model_output = model(model_features, model_target)
prediction = recipe.target.inverse_transform(model_output)
prediction = recipe.output.transform(prediction)
```

There is no public `ClassShuffle.correct_output` escape hatch and callers do
not reach into `recipe.target.steps`.

`recipe.output` processors must use the same head-aware container, or an
explicit adapter must materialize their expected table representation.
`SoftmaxTemperature`, for example, acts on a class-score head without removing
its target ownership or class-axis metadata.

For backward compatibility, `inverse_transform(TableTensor)` can be adapted as
`OutputKind.POINT` when its columns match the recorded transformed target
schema. Models should produce a typed `ModelOutput` internally. Public model
APIs may continue returning tensors while a model-owned adapter attaches the
output specification before recipe processing.

The inverse result retains output representation. Class scores remain scores,
quantiles remain quantiles, and point outputs remain points. The result gains
the original target ordering and metadata; decoding an argmax into category
values is a separate terminal operation.

## `ClassShuffle`

For one categorical target, let the fitted permutation `P` map an original
class index to the shuffled index used for training:

```text
shuffled_label = P[original_label]
```

### Point predictions

A hard prediction in shuffled coordinates is mapped with the inverse
permutation:

```python
original_label = P.argsort()[shuffled_label]
```

Missing negative codes remain unchanged when point outputs use categorical
code conventions.

### Class-score predictions

Let `scores_z[..., j]` be the model score for shuffled class `j`. Scores in
original order are:

```python
scores_y = scores_z.index_select(class_axis, P)
```

For example, with `P = [2, 0, 1]`:

```text
original class:       0      1      2
shuffled class:       2      0      1
scores in Z order:  [z0,    z1,    z2]
scores in Y order:  [z2,    z0,    z1]
```

The operation applies equally to logits and probabilities. `ClassShuffle`
does not apply softmax, argmax, or category decoding. It only aligns the class
axis and restores the fitted original category metadata.

The implementation validates that the event-axis size equals the fitted class
count. For multiple categorical targets, each target has its own head and its
own slice of the flattened permutation buffer.

### Why point and score inverses differ

The point inverse answers, "which original class produced shuffled class
`j`?" The score inverse answers, "where is the score for original class `i`
stored?" The former indexes by `P^-1`; the latter gathers by `P`. A generic
mathematical inverse over label tensors cannot silently double as a class-axis
inverse.

## Composition

### `Sequential`

Target processors run in reverse order. Each step receives typed heads plus the
fitted target context. A step must either support the head's `OutputKind` or
raise a clear error.

Numerical bijections such as `StandardScale`, `Power`, and `Quantile` apply
their inverse to point, quantile, sample, or other support-value heads. They do
not transform logits or unrelated distribution parameters merely because the
payload is floating point.

### `StypeDispatch`

During target fitting, dispatch records which route owns each target and how
the route changes the target schema. During inverse processing, it groups
model-output heads by that recorded ownership and delegates whole heads to the
route processor.

Forward widths from PR #202 remain useful as a fallback layout for structured
point outputs. They are not model-head boundaries. In particular:

```text
categorical target width:       1
classification output width:    C

numerical target width:         1
TabICLv2 quantile output width:  999
```

Slicing either model output with a recorded width of one would be incorrect.
Explicit `OutputHead` boundaries solve this without relying on names or
ordering conventions.

### `Choice` and task dispatch

`Choice` applies the inverse of the option selected during fit. Task dispatch
applies the inverse for the selected task. The selected branch and fitted
target schema belong to each estimator's processing state.

## Fitted state and serialization

Each estimator's target pipeline state includes processor parameters, original
and transformed target schemas, route ancestry, and the selected task or choice
branch. The model-output specification is model configuration; resolved head
ownership is fitted recipe state.

This design does not add a `ClassShuffle`-specific checkpoint workaround.
Serializing dynamically sized fitted buffers, `_fitted`, schemas, and composite
selection is a processor-wide concern and should be solved consistently for the
whole processing package.

## Ensemble boundary

Inverse target processing happens per estimator before ensemble aggregation:

```text
for each estimator:
    transform that estimator's target
    run the model
    attach the model-output layout
    inverse-transform the model output into original target coordinates

aggregate aligned estimator outputs
apply recipe.output postprocessing
```

This ordering is required when ensemble members draw different class
permutations. Averaging unaligned class scores mixes different classes.
Nonlinear numerical inverse transforms may also fail to commute with averaging.

The existing model loop averages raw estimator tensors. Recipe integration
must move target inversion inside that loop or guarantee that every estimator
shares identical target-processing state. The former is the general rule.

`recipe.output` remains post-inverse cleanup. Whether an ensemble averages
logits or probabilities is a model-level decision and must be explicit; it is
not inferred by processors.

## Rejected alternatives

### Public processor-specific output correction

A method such as `ClassShuffle.correct_output(tensor)` exposes pipeline
internals, does not compose through `Sequential`, `Choice`, or dispatch, and
forces callers to locate fitted steps. The output inverse belongs to the target
pipeline.

### Infer heads from forward widths

Forward widths describe transformed target tables, not prediction heads. They
cannot distinguish a one-column categorical target with `C` logits from a
one-column numerical target with `Q` quantiles. Width inference is retained
only as a point-output compatibility path.

### Infer semantics from dtype or rank

Floating tensors may be logits, probabilities, quantiles, samples, point
regression values, or opaque distribution parameters. Tensor rank also varies
with batch shape. Silent inference would apply valid operations to the wrong
semantic object.

### Require only a `TableTensor` model output

Column names and stypes still do not say whether a numerical block is a set of
targets, class scores, or quantiles. A table can be the values payload, but it
does not replace an explicit head kind and target reference.

## Validation and errors

The target pipeline should reject:

- inverse processing before target fitting;
- raw tensors without an unambiguous model-output specification;
- missing, duplicated, or unknown target ownership;
- unsupported output kinds for any processor in the reverse chain;
- class heads whose event-axis size differs from fitted cardinality;
- model-output layouts that omit or reorder targets without declaring it.

It should allow arbitrary leading batch and row dimensions and preserve device
and dtype. It must not compare prediction row count with the fitted target row
count.

## Proposed implementation sequence

1. Introduce `OutputKind`, `OutputHead`, `ModelOutput`, and fitted target-schema
   records without changing model return types.
2. Specialize `Recipe.target` into a target pipeline that adapts model outputs
   and reverses processors.
3. Implement head-aware inverses for existing numerical processors and
   `ClassShuffle`.
4. Extend `Sequential`, `Choice`, task dispatch, and `StypeDispatch` to route
   typed heads by fitted target ancestry. Coordinate this step with PR #202.
5. Move target inversion inside the per-estimator model loop, then aggregate
   aligned outputs and run `recipe.output`.
6. Add multi-target adapters only after model output layouts are explicit.

## Required tests

- `ClassShuffle` hard predictions use `P^-1`.
- `ClassShuffle` logits and probabilities gather the class axis with `P` on CPU
  and CUDA while preserving arbitrary leading dimensions.
- Two estimators with different permutations are aligned before averaging.
- `StandardScale` inverses point and multi-quantile heads by broadcasting over
  the value axis.
- `Sequential` applies mixed target inverses in reverse order.
- `StypeDispatch` routes a width-one categorical target to a width-`C` score
  head and a width-one numerical target to a width-`Q` quantile head.
- Ambiguous raw multi-target tensors and unsupported output kinds fail clearly.
- Output postprocessing runs after inverse alignment and ensemble aggregation.

## Open naming decisions

- `ModelOutput` versus `Prediction` for the typed container.
- `OutputKind` versus a protocol implemented by concrete output-head types.
- Whether the raw-tensor adapter lives on the model, the target pipeline, or a
  small object shared by both. The model must remain the source of output
  semantics in every variant.

These naming choices do not change the core contract: model outputs carry head
semantics, target processors implement the induced inverse for supported head
kinds, and composition routes heads by fitted target ownership.
