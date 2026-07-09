# Model-output inverse processing design

## Status and decision

This document defines the smallest inverse-processing contract needed by
TabICLv2. It is a design and an executable validation, not an implementation.

The MVP makes the following decisions:

- keep `Recipe.target` as an ordinary `Processor` or `Sequential`; do not add
  a `TargetPipeline`;
- use the existing `inverse_transform` API for raw model outputs; do not add a
  second `inverse_transform_output` API;
- do not require an output specification such as `Quantiles(width=999)`;
- apply `ClassShuffle.inverse_transform` to class scores, before softmax or
  argmax; do not implement a hard-class-ID inverse for this path;
- support the single-target output contract that TabICLv2 already exposes;
- introduce typed model outputs only when a model needs multiple or ambiguous
  heads.

This is narrower than a universal model-output abstraction, but it covers the
current TabICL regression and classification paths without speculative API.

## Repository facts that define the MVP

The existing API already assigns inverse processing to the target role:

```python
model_target = recipe.target.fit_transform(target)
raw_output = model(model_features, model_target)
prediction = recipe.target.inverse_transform(raw_output)
```

`Recipe.target` is currently a plain processor, normalized to `Sequential`
when a list is provided. Therefore a new pipeline type is not necessary for
composition or task dispatch.

TabICLv2 has a single target and returns a raw `torch.Tensor`:

- integer targets select classification and produce a fixed-width head of 10
  logits;
- floating-point targets select regression and produce 999 quantiles;
- the model does not return class IDs. IDs only appear later if a prediction
  driver calls `argmax`.

The target processor is already fitted when the model is called. Its fitted
state supplies the missing information required by the two supported inverse
operations: the regression mean and scale, or the active class count and
class permutation.

## Minimal contract

`inverse_transform` continues to mean: map an object expressed in transformed
target coordinates back to original target coordinates. The accepted
representation is broadened from only `TableTensor` to include a raw model
output `Tensor`:

```python
def inverse_transform(
    self,
    input: TableTensor | Tensor,
) -> TableTensor | Tensor:
    self._check_is_fitted()
    return self._inverse_transform(input)
```

Existing `TableTensor` inverse behavior remains valid. When the input is a raw
`Tensor`, the processor is being called in `Recipe.target` on a model output.
The fitted processor and the single-target TabICL contract determine the
operation; dtype, rank, and generated column names are not inspected.

A `ClassShuffle` used on a categorical feature block still transforms every
categorical feature independently. Feature preprocessing does not call its
inverse on model predictions. In the MVP, the raw-output inverse is supported
only when that instance was fitted on the single categorical target.

### Regression: `StandardScale`

For a single numerical target with fitted mean `m` and scale `s`, every value
emitted by the regression head is in the same transformed target coordinate
system:

```python
restored = output * scale[0] + mean[0]
```

The operation broadcasts over point predictions, samples, or any number of
quantiles. It does not need to know whether the last dimension has width 1,
17, or 999. Consequently this is sufficient:

```python
Recipe(target=StandardScale())
```

This is unnecessary duplication and is not part of the MVP:

```python
TargetPipeline(
    StandardScale(),
    output_spec=Quantiles(width=999),
)
```

The value 999 is a TabICLv2 architecture contract and remains validated in
the model tests. It is not target-recipe configuration.

`StandardScale` keeps its existing `TableTensor` branch for data round trips
and adds the raw-tensor branch using the same fitted buffers:

```python
def _inverse_transform(self, input: TableTensor | Tensor):
    if isinstance(input, TableTensor):
        numerical = _as_float(input.numerical) * self.scale + self.mean
        return input.replace_blocks(numerical=numerical)
    return input * self.scale[0] + self.mean[0]
```

The scalar indexing is intentional: TabICLv2 supports one target while its
regression output has a value/support axis of arbitrary width.

### Classification: `ClassShuffle`

Let the fitted permutation `P` map original class codes to the shuffled codes
shown to the model:

```text
shuffled_code = P[original_code]
```

The model returns scores in shuffled-code order. The score for original class
`i` is therefore stored at position `P[i]`, so original class order is
restored with:

```python
scores_original = scores_shuffled.index_select(-1, P)
```

TabICLv2 always emits 10 logits even when only `K < 10` classes are active.
The fitted category count supplies `K`; the processor first selects the active
head prefix and then restores the original class order:

```python
def _inverse_transform(self, output: Tensor) -> Tensor:
    # MVP: exactly one fitted categorical target.
    stop = self.offsets[1]
    permutation = self.permutations[:stop]
    return output[..., :stop].index_select(-1, permutation)
```

There is deliberately no equality check between fitted class count and model
head width. A five-class target is valid for TabICL's ten-wide head. If the
head is shorter than the required permutation, the tensor indexing operation
fails rather than introducing a second, inconsistent width contract.

This operation is the same for logits and probabilities, but for TabICL it
must run on logits before output postprocessing. In particular, softmax over
all ten logits would incorrectly include inactive classes. The intended flow
is:

```text
10 raw logits
  -> select K active logits
  -> restore original class order
  -> align/aggregate estimators
  -> output processing such as temperature and softmax
  -> optional argmax and category decoding
```

No class-ID case is implemented. Supporting an external model whose public
`predict` method returns IDs would be a different model-output contract. It
must not be guessed from the values or shape of a tensor.

The original target's category vector remains the terminal decoding metadata.
`ClassShuffle.inverse_transform` only aligns the score axis; it does not apply
softmax, argmax, or decode category values.

## Composition without `TargetPipeline`

### `Sequential`

The existing reverse traversal is sufficient. Its annotations need to admit
raw tensors, and every step in a target sequence must support the
representation it receives:

```python
def _inverse_transform(self, output: TableTensor | Tensor):
    for step in reversed(self.steps):
        output = step.inverse_transform(output)
    return output
```

### Task and choice dispatch

Task dispatch records the branch selected while fitting the target and
delegates `inverse_transform` to that same branch. It does not infer a task
from a model-output tensor:

```python
target=TaskDispatch(
    classification=ClassShuffle(),
    regression=StandardScale(),
)
```

`Choice` follows the same rule: inverse processing uses the choice selected
during fit. This gives the intended task-dispatch flow without adding an
output specification or specialized target container.

For a single target, `StypeDispatch` can likewise delegate the entire raw
output to the one fitted target route. The width-aware `TableTensor` inverse
proposed in [PR #202](https://github.com/NVIDIA/structured-data-models/pull/202)
remains useful for structured table inverses, but its recorded input widths
must not slice TabICL's raw 10- or 999-wide model heads.

## Ensemble boundary

With the current recipe state, all estimators share one fitted target
processor. Both MVP operations are affine/linear in the output, so applying
the shared inverse immediately after the current averaged model output is
equivalent to applying it to every member first.

If estimators later fit independent `ClassShuffle` permutations, each member
must be aligned before averaging:

```python
aligned = []
for estimator, target_processor in estimators:
    raw = estimator(...)
    aligned.append(target_processor.inverse_transform(raw))
prediction = torch.stack(aligned).mean(dim=0)
```

Averaging scores that use different class-coordinate systems mixes unrelated
classes. This ordering requirement does not imply a `TargetPipeline`; it only
requires the model/driver to call the already fitted processor at the correct
boundary.

## Executable evidence

The proposal was exercised on CPU against repository commit `ec91baa` using
the real `TabICLv2(pretrained=False)`, `ClassShuffle.fit`, and
`StandardScale.fit` implementations. The controlled score tensors isolate
coordinate correctness; this is not a model-quality benchmark.

The run used two independently fitted five-class permutations and 4,096 rows
inside TabICL's fixed ten-wide classification head:

| Measurement                                       |            Result |
| ------------------------------------------------- | ----------------: |
| Actual TabICL classification output shape         |         `[2, 10]` |
| Actual TabICL regression output shape             |        `[2, 999]` |
| Permutation A                                     | `[4, 2, 0, 3, 1]` |
| Permutation B                                     | `[0, 4, 2, 1, 3]` |
| Maximum score error after per-member alignment    |             `0.0` |
| Argmax mismatch after aligned aggregation         |            `0.0%` |
| Argmax mismatch after naive unaligned aggregation |        `71.2891%` |

The naive mismatch is evidence for the ensemble boundary: the same class
evidence becomes wrong when different shuffled axes are averaged without
first gathering each axis by its own fitted permutation.

The regression experiment fitted `StandardScale` to 257 target values and
round-tripped random raw outputs of three widths:

| Output width | Shape preserved | Maximum round-trip error |
| -----------: | :-------------: | -----------------------: |
|            1 |       yes       |                `1.49e-7` |
|           17 |       yes       |                `3.58e-7` |
|          999 |       yes       |                `4.77e-7` |

This demonstrates that the same fitted scalar state handles 999 quantiles
without a `Quantiles(width=999)` declaration.

The following script is self-contained when run from the repository with
`uv run python`:

```python
import torch

from sdm import CategoricalTensor, StringTensor, TableTensor
from sdm.models import TabICLv2
from sdm.processing import ClassShuffle, StandardScale

torch.manual_seed(20260709)
model = TabICLv2(pretrained=False)
x = torch.randn(6, 4)
assert model(x, torch.tensor([0, 1, 2, 3])).shape == (2, 10)
assert model(x, torch.tensor([1.0, 2.0, 4.0, 8.0])).shape == (2, 999)

target = TableTensor(
    columns={"categorical": ("target",)},
    categorical=CategoricalTensor(
        data=(torch.arange(256, dtype=torch.int32) % 5).unsqueeze(-1),
        categories=(
            StringTensor.from_list(["a", "b", "c", "d", "e"]),
        ),
    ),
)

def fit_shuffle(seed):
    torch.manual_seed(seed)
    return ClassShuffle(method="random").fit(target)

shuffle_a, shuffle_b = fit_shuffle(17), fit_shuffle(29)
p_a, p_b = shuffle_a.permutations, shuffle_b.permutations
k = int(shuffle_a.offsets[1])

torch.manual_seed(1234)
base = torch.randn(4096, k)
original_a = base + 0.15 * torch.randn_like(base)
original_b = base + 0.15 * torch.randn_like(base)
raw_a, raw_b = torch.randn(4096, 10), torch.randn(4096, 10)
raw_a[..., :k] = original_a.index_select(-1, p_a.argsort())
raw_b[..., :k] = original_b.index_select(-1, p_b.argsort())

aligned_a = raw_a[..., :k].index_select(-1, p_a)
aligned_b = raw_b[..., :k].index_select(-1, p_b)
aligned = torch.stack([aligned_a, aligned_b]).mean(0)
expected = torch.stack([original_a, original_b]).mean(0)
naive = torch.stack([raw_a[..., :k], raw_b[..., :k]]).mean(0)

torch.testing.assert_close(
    aligned,
    expected,
    rtol=0,
    atol=0,
)
print((naive.argmax(-1) != expected.argmax(-1)).float().mean())
print((aligned.argmax(-1) != expected.argmax(-1)).float().mean())

torch.manual_seed(4321)
values = 11.2 + 3.7 * torch.randn(257, 1)
scale = StandardScale().fit(TableTensor.from_tensor(values))
for width in (1, 17, 999):
    raw = torch.randn(32, width)
    restored = raw * scale.scale[0] + scale.mean[0]
    round_trip = (restored - scale.mean[0]) / scale.scale[0]
    assert restored.shape == raw.shape
    print(width, (round_trip - raw).abs().max())
```

The first two printed values are:

```text
tensor(0.7129)
tensor(0.)
```

The three maximum regression round-trip errors are:

```text
1   tensor(1.4901e-07)
17  tensor(3.5763e-07)
999 tensor(4.7684e-07)
```

## Minimal implementation sequence

1. Broaden `InvertibleMixin` and `Sequential` inverse annotations to accept a
   raw `Tensor` in addition to `TableTensor`.
2. Add the width-agnostic single-target tensor branch to `StandardScale` while
   preserving its existing table inverse.
3. Make `ClassShuffle` invertible for a raw class-score tensor fitted on one
   categorical target: select the active prefix and gather it by the fitted
   permutation.
4. Have task/choice dispatch delegate raw-output inverse processing to the
   branch selected during target fitting.
5. Call `recipe.target.inverse_transform(raw_output)` before output
   postprocessing and argmax. Move this call inside estimator aggregation when
   estimator-specific target states are introduced.
6. Keep the fixed widths 10 and 999 in TabICLv2 model tests, not in recipes.

## Required tests

- `StandardScale.inverse_transform(Tensor)` preserves shapes and broadcasts
  the single fitted mean/scale over a 999-wide output.
- Existing `StandardScale.inverse_transform(TableTensor)` round trips remain
  unchanged.
- `ClassShuffle.inverse_transform(Tensor)` accepts a ten-wide head with
  `K < 10`, returns width `K`, and gathers the last axis by `P` while
  preserving arbitrary leading dimensions, dtype, and device.
- Classification inversion runs before softmax and argmax; there is no
  class-ID inverse test in the MVP.
- Two independently shuffled estimator outputs are aligned before aggregation
  and reproduce aggregation in original class order.
- `Sequential` applies raw-output inverses in reverse order.
- Task dispatch uses its fitted classification or regression branch without
  inspecting the raw output.
- Existing multi-column categorical feature tests continue to prove that
  `ClassShuffle.transform` works on feature blocks.

## Deliberate limitations and generalization trigger

The MVP rejects or defers:

- multiple target heads in one raw tensor;
- coupled multi-target outputs;
- distribution parameters whose inverse is not elementwise;
- externally produced hard class IDs;
- per-head axis declarations other than TabICL's last-axis convention;
- reconstructing a feature `TableTensor` through `ClassShuffle`.

When one of these becomes a concrete model requirement, a typed `ModelOutput`
with explicit head ownership, output kind, and event axis becomes justified.
That is the point at which a specialized target pipeline may add value. It is
not required to make the current TabICL task-dispatch flow work.

Serialization of fitted buffers, schemas, and selected composite branches
remains a processing-package-wide issue. This design does not add a
`ClassShuffle`-specific state-loading workaround.
