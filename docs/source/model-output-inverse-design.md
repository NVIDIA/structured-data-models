# Model-output inverse processing design

## Status and MVP decision

This document defines the smallest model-output inverse contract needed by
TabICLv2. It combines the implemented `ClassShuffle` target inverse with the
proposed model/ensemble integration needed to apply it.

The MVP makes the following decisions:

- target inverse processing is in scope and must work for every ensemble
  member;
- keep the existing `TableTensor -> TableTensor` processor and
  `inverse_transform` contract;
- wrap each raw member output in a transient numerical `TableTensor` before
  calling its fitted target inverse;
- align every member in original target coordinates before averaging;
- run user output processors, such as temperature scaling and softmax, once
  after averaging aligned model outputs;
- keep `Recipe.target` as an ordinary `Processor` or `Sequential`; do not add
  `TargetPipeline`;
- do not add `inverse_transform_output` or an output declaration such as
  `Quantiles(width=999)`;
- defer `LabelDecode`, preservation of original category metadata, categorical
  model returns, and the final public `TableTensor` return contract.

While terminal decoding is deferred, the recipe-integrated model may preserve
the existing `Tensor` return API by returning the numerical block after output
processing. The intermediate target inverse still uses `TableTensor` and is
fully compositional with existing processors.

This change implements and tests
`ClassShuffle.inverse_transform(TableTensor)`. Creating and storing one
fitted recipe per ensemble member, invoking target inverse inside the model
loop, and applying user output after aggregation remain follow-up integration
work.

## Current model-output contracts

At repository commit `ec91baa`, `BaseModel` collects raw tensors from
`_forward` and averages them. TabICLv2 chooses its head from the target dtype:

| Target              | Raw member output    | Meaning                                 |
| ------------------- | -------------------- | --------------------------------------- |
| Integer/categorical | `[..., R_test, 10]`  | Unactivated class logits                |
| Floating/numerical  | `[..., R_test, 999]` | Quantiles at levels 0.001 through 0.999 |

Both heads end in a linear layer. The classifier does not apply softmax, and
the regressor does not wrap or annotate its quantile axis.

KumoRFM currently returns a placeholder ten-wide tensor and does not yet
declare whether it represents logits, probabilities, or another output kind.
This design therefore makes no KumoRFM-specific postprocessing claim.

## Required ordering

The model/recipe boundary is:

```text
for each ensemble member:
    independently fit/copy that member's recipe
    preprocess that member's inputs and target
    run the model
    wrap the raw output as a numerical TableTensor
    apply that member's fitted target inverse

validate that aligned member outputs have a common numerical layout
average aligned numerical values
apply user output processors once
return the numerical result for the MVP
```

In compact form:

```text
member model output
  -> member target inverse
  -> average aligned outputs
  -> user output processing
```

### Why target inverse is per member

Independent recipes may draw different `ClassShuffle` permutations or select
different fitted target transforms. Raw outputs then use different target
coordinate systems and cannot be averaged directly.

For classification, every member must first map its fixed ten-wide head into
the original active-class order. For regression, every quantile must first be
mapped back to original target units. `StandardScale` happens to commute with
averaging when every member has identical fitted state, but that special case
must not define the general boundary.

### Why user output processing is after averaging

The current `BaseModel` architecture defines an ensemble over raw head values.
For classification, this means averaging aligned logits and applying the
configured activation afterwards:

```python
mean_logits = torch.stack(aligned_logits).mean(dim=0)
probabilities = torch.softmax(mean_logits / temperature, dim=-1)
```

This intentionally differs from averaging per-member probabilities:

```python
torch.stack(
    [torch.softmax(logits / temperature, dim=-1) for logits in aligned_logits]
).mean(dim=0)
```

Softmax is nonlinear, so these are different ensemble definitions. The MVP
chooses the first definition and must test it explicitly. A future model may
declare another aggregation contract rather than silently changing processor
placement.

## `TableTensor` boundary

The adapter introduced by
[PR #221](https://github.com/NVIDIA/structured-data-models/pull/221) is useful
but currently placed after averaging. The wrap must happen inside the
estimator loop so target inversion can align each member first.

The intended integration is:

```python
aligned: list[TableTensor] = []
fitted_recipes: list[Recipe] = []

for _ in range(num_estimators):
    recipe_i = copy.deepcopy(recipe)
    x_i, y_i = fit_preprocess(recipe_i, x, y)
    raw_i = self._forward(x_i, y_i, related_tables, cache=None)

    # TableTensor operations currently need to leave inference mode.
    with torch.inference_mode(False):
        table_i = TableTensor.from_tensor(raw_i.clone())
        table_i = recipe_i.target.inverse_transform(table_i)

    aligned.append(table_i)
    fitted_recipes.append(recipe_i)

values = torch.stack([table.numerical for table in aligned]).mean(dim=0)
mean_table = TableTensor.from_tensor(values)

# All members resolve the same task. Use one fitted recipe so TaskDispatch has
# its resolved branch.
with torch.inference_mode(False):
    output = fitted_recipes[0].output.transform(mean_table)

return output.numerical
```

For `fit`/`predict`, each cache must retain its corresponding fitted recipe.
Prediction transforms test features and inverses model output with the recipe
paired to that cache. A single shared recipe, as currently sketched in PR
#221, cannot represent independent ensemble preprocessing.

The transient `TableTensor` contains numerical score or quantile columns. It
does not claim that those columns are independent original target columns, and
it does not carry category metadata. Those semantics are unnecessary for the
MVP target inverse.

## Regression: `StandardScale`

For a single target with fitted mean `m` and scale `s`, all 999 TabICL
quantiles are expressed in the same transformed coordinate system. The
existing implementation already broadcasts its one-element fitted buffers
over any numerical output width:

```python
def _inverse_transform(self, input: TableTensor) -> TableTensor:
    numerical = _as_float(input.numerical) * self.scale + self.mean
    return input.replace_blocks(numerical=numerical)
```

Therefore the target recipe remains:

```python
Recipe(target=StandardScale())
```

The value 999 is a TabICLv2 head contract, already validated by model tests.
The processor does not need `Quantiles(width=999)` and must not compare the
prediction row count with the target-fitting row count.

## Classification: `ClassShuffle`

Let the fitted permutation `P` map an original class code to the shuffled code
used as model context:

```text
shuffled_code = P[original_code]
```

The raw score for original class `i` is stored at shuffled position `P[i]`.
The target inverse must therefore gather the score axis with `P`:

```python
scores_original = scores_shuffled.index_select(-1, P)
```

TabICLv2 always emits 10 logits, while a fitted target may have `K < 10`
categories. The inverse selects the active prefix and returns an aligned
numerical `TableTensor` of width `K`:

```python
class ClassShuffle(Processor, InvertibleMixin):
    def _inverse_transform(self, input: TableTensor) -> TableTensor:
        # MVP: one fitted categorical target.
        num_classes = int(self.offsets[1])
        permutation = self.permutations[:num_classes]
        scores = input.numerical[..., :num_classes].index_select(
            -1,
            permutation,
        )
        return TableTensor.from_tensor(scores)
```

The implementation constructs a new table because `TableTensor` deliberately
rejects positional slicing of its column dimension. Slicing and gathering are
performed on `input.numerical` instead.

There is no equality check between fitted category count and model head width:
a five-class target is valid for TabICL's ten-wide head. If the head is shorter
than required, tensor indexing fails. The inverse preserves arbitrary leading
dimensions, dtype, and device.

This operation does not apply softmax, argmax, or category decoding. The model
does not emit hard class IDs, so an ID inverse is not part of the MVP.

## Composition

### `Sequential`

No new sequence type is needed. The existing inverse traversal already passes
a `TableTensor` through invertible target steps in reverse order:

```python
def _inverse_transform(self, input: TableTensor) -> TableTensor:
    output = input
    for step in reversed(self.steps):
        output = step.inverse_transform(output)
    return output
```

### Task and choice dispatch

Output `TaskDispatch` is resolved while the member recipe fits its target.
After aligned member outputs are averaged, output processing uses one fitted
member recipe and delegates to its resolved route:

```python
TaskDispatch(
    classification=SoftmaxTemperature(temperature=0.9),
    regression=Identity(),
)
```

`Choice` target inversion delegates to the option selected during that
member's fit. A composite processor must use fitted branch/route state rather
than infer a route from the numerical model-output table.

The width-aware structured `StypeDispatch` inverse proposed in
[PR #202](https://github.com/NVIDIA/structured-data-models/pull/202) remains
useful for ordinary table round trips. Its recorded forward target width of
one is not a model-head boundary and must not slice a ten-logit or 999-quantile
head. The raw single-target model-output path must delegate the complete head
to the fitted target route.

## Executable evidence

The design was exercised on CPU against `ec91baa` using the real
`TabICLv2(pretrained=False)`, the implemented `ClassShuffle` target
inverse, `StandardScale.inverse_transform`, and existing `TableTensor`
operations.

The controlled classification experiment used two independently fitted
five-class permutations and 4,096 rows:

| Measurement                                          |                  Result |
| ---------------------------------------------------- | ----------------------: |
| Actual TabICL classification output shape            |               `[2, 10]` |
| Actual TabICL regression output shape                |              `[2, 999]` |
| Permutation A                                        |       `[4, 2, 0, 3, 1]` |
| Permutation B                                        |       `[0, 4, 2, 1, 3]` |
| Target-inverse output type and shape                 | `TableTensor [4096, 5]` |
| Maximum score error after per-member target inverse  |                   `0.0` |
| Argmax mismatch after aligned aggregation            |                  `0.0%` |
| Argmax mismatch after naive unaligned aggregation    |              `71.2891%` |
| Maximum difference between the two softmax orderings |             `0.0162606` |

The last value confirms that post-average softmax and averaging member
probabilities are measurably different contracts. The MVP selects
post-average softmax.

The regression experiment fitted the actual `StandardScale` processor to 257
target values, called its existing `inverse_transform(TableTensor)`, and
round-tripped three output widths:

| Output width | Shape preserved | Maximum round-trip error |
| -----------: | :-------------: | -----------------------: |
|            1 |       yes       |                `1.49e-7` |
|           17 |       yes       |                `3.58e-7` |
|          999 |       yes       |                `4.77e-7` |

This validates that the existing fitted scalar state handles 999 quantiles
without any output-width declaration.

The essential reproducible checks are:

```python
table_a = class_shuffle_a.inverse_transform(
    TableTensor.from_tensor(raw_a)
)
table_b = class_shuffle_b.inverse_transform(
    TableTensor.from_tensor(raw_b)
)
mean_table = TableTensor.from_tensor(
    torch.stack([table_a.numerical, table_b.numerical]).mean(0)
)
torch.testing.assert_close(mean_table.numerical, expected_original_scores)

# Existing StandardScale inverse, with a 999-wide model output.
raw = TableTensor.from_tensor(torch.randn(32, 999))
restored = standard_scale.inverse_transform(raw)
round_trip = standard_scale.transform(restored)
torch.testing.assert_close(round_trip.numerical, raw.numerical)
```

## Remaining model integration sequence

1. Create one deep-copied and independently fitted recipe per ensemble member.
2. Pair every cached estimator state with its fitted recipe.
3. Wrap each raw member output as a transient numerical `TableTensor` before
   target inversion.
4. Call the fitted target inverse for each member. `ClassShuffle` and
   `StandardScale` provide the required single-target operations.
5. Validate compatible aligned numerical shapes and average their values.
6. Run the resolved user output pipeline once after averaging.
7. Preserve the current numerical `Tensor` return while decoding and final
   structured return semantics remain deferred.

## Required tests

- `StandardScale.inverse_transform(TableTensor)` broadcasts one fitted
  mean/scale over a 999-wide model output on CPU and CUDA.
- Existing one-column `StandardScale` target round trips remain unchanged.
- `ClassShuffle.inverse_transform(TableTensor)` accepts a ten-wide numerical
  score table with `K < 10`, returns a width-`K` numerical `TableTensor`, and
  gathers its last axis with `P`.
- `ClassShuffle` target inverse preserves arbitrary leading dimensions, dtype,
  and device.
- Two members with different permutations are inversed before aggregation and
  reproduce aggregation in original class order.
- The output activation runs exactly once after aggregation and matches
  `softmax(mean(aligned_logits) / temperature)`.
- Forward and cached prediction pair each member with its own fitted recipe.
- `Sequential` applies target inverses in reverse order.
- Task dispatch uses its fitted route without inspecting the model-output
  table.
- Existing multi-column feature tests continue to prove that
  `ClassShuffle.transform` works on categorical feature blocks.

## Deferred problems

The following are deliberately future work:

- `LabelDecode` and other terminal representation-changing processors;
- retention and serialization of the original target name and category
  vectors;
- returning a categorical `TableTensor` from `BaseModel`;
- a public probability object that retains category metadata;
- externally produced hard class IDs;
- multiple or coupled target heads;
- model outputs whose inverse depends on non-value distribution parameters;
- a KumoRFM output contract.

When terminal decoding becomes a requirement, the original target metadata
must be captured before target transformation. That future decision may add a
target context or typed prediction carrier. It is not required for correct
target inversion and aggregation now.

Serialization of dynamically sized fitted buffers and composite selection
remains a processing-package-wide concern. This design adds no
`ClassShuffle`-specific state-loading workaround.
