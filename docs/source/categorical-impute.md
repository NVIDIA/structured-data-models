# Categorical alignment and imputation design

Status: implemented by {py:class}`~sdm.processing.CategoricalAlign` and
{py:class}`~sdm.processing.CategoricalImpute`.

## Context

{py:class}`~sdm.tensor.CategoricalTensor` stores integer codes together with a
category vocabulary per column. Its dataframe constructors infer that
vocabulary from the input being converted, and negative codes represent
missing values. Independently tensorized training and query tables may
therefore assign different codes to the same value. Tensorizing them jointly
before slicing gives them one vocabulary, but that vocabulary can contain
query-only values.

The missing operation is not ordinal encoding: the codes already exist. It is
alignment to a vocabulary learned from the training or context rows. Unknown
query values must become the same negative sentinel used for missing values.
Recipes that require fully observed categorical inputs can then impute that
sentinel from training state.

[PR #230](https://github.com/NVIDIA/structured-data-models/pull/230/files)
shows the intended composition point: categorical features are routed through
`ToNumerical` before the shared numerical pipeline. The categorical route can
instead be written as:

```python
StypeDispatch(
    categorical=[
        CategoricalAlign(),
        CategoricalImpute(strategy="most_frequent"),
        ToNumerical(),
    ],
)
```

The recipe fits this route on context or training rows and reuses its fitted
state for query rows. Query values influence neither the fitted vocabulary nor
the selected fill category.

## Decision

Add two dedicated categorical processors with separate responsibilities.

### Vocabulary alignment

```python
CategoricalAlign()
```

During `fit`, the processor stores only category values referenced by
non-negative codes in the fitted rows. This removes unused metadata, including
query-only categories from a jointly tensorized table. Fitted category order
follows first occurrence in the fitted rows and therefore does not inherit
query ordering from a joint vocabulary. During `transform`, it maps the
input's local codes to the fitted codes by category value. Missing and unknown
values become `-1`, and the output carries the fitted vocabularies. Column names
and order must match the fitted schema. Category value types and numeric dtypes
must also match unless the fitted vocabulary is empty; complex category values
are unsupported. Non-negative codes outside their declared input vocabulary
are invalid and rejected.

### Missing-value imputation

```python
CategoricalImpute(strategy="most_frequent")
```

During `fit`, the processor counts non-negative category codes independently
for each column. During `transform`, it replaces every negative code with the
fitted mode. Ties select the lowest code, making the result deterministic. The
operation preserves column order, shape, integer dtype, device, and category
vocabularies. Fitting fails when a column has no observed value because its
mode is undefined. A non-negative code outside its declared vocabulary is
invalid categorical data and is rejected during both fitting and
transformation.

The imputation strategy matches KumoRFM's `MOST_FREQUENT` behavior without assuming
that category vocabularies are already frequency-sorted. It also provides an
explicit opt-in before the `-1` sentinel from TabICL-style encoding crosses
into a numerical-only model pipeline.

## Why separate processors

- Alignment changes code and vocabulary semantics but does not choose a
  missing-value policy. Models that accept `-1`, such as TabICL, can use
  `CategoricalAlign` without imputation.

- Imputation learns a distribution-dependent fill value and requires aligned
  vocabularies. Combining both operations would force every consumer to impute
  unknown values and would hide the two fitted states.

- Extending `MeanImpute` would mix numerical NaN reduction with categorical
  code and vocabulary semantics.

- Extending `ToNumerical` would make an stype conversion unexpectedly learn
  state and would hide the context-only fitting boundary.

- Adding a separate category would mutate vocabularies and may change model
  embedding sizes; that is a distinct future strategy.

- `ClassShuffle` permutes category codes for ensemble diversity and preserves
  missing values. Imputation has different state and output semantics.

The `strategy` argument is intentionally narrow. Future strategies such as a
constant or separate category can extend the same public class once their
vocabulary and unseen-value contracts are defined.

## Evidence

- [TabICL preprocessing](https://github.com/soda-inria/tabicl/blob/46b91961db4f8873dd049ec09990698a435e1e29/src/tabicl/_sklearn/preprocessing.py)
  encodes missing and unknown categorical values as `-1` while imputing only
  numerical columns.
- [PyTorch Frame categorical mapping](https://github.com/pyg-team/pytorch-frame/blob/b79012c111f8bbe38f0f10897df9fb08ceaf7725/torch_frame/data/mapper.py#L85-L115)
  maps values against a supplied vocabulary and encodes missing or unseen
  values as `-1`.
- [KumoRFM categorical encoding](https://github.com/kumo-ai/kumo-ml/blob/6cb93eff313fbf770d33183098fe3ed2ec55accf/kumoml/common/nn/fused_encoder/fused_categorical.py#L146)
  maps missing values to the most frequent category for its
  `MOST_FREQUENT` strategy.
