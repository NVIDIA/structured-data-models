# Categorical imputation design

Status: implemented by {py:class}`~sdm.processing.CategoricalImpute`.

## Context

Categorical missing values in {py:class}`~sdm.tensor.CategoricalTensor` use a
negative integer sentinel. {py:class}`~sdm.processing.ToNumerical` deliberately
preserves that sentinel when it moves category codes into the numerical block.
This is faithful to TabICL's `OrdinalEncoder`, but it leaves recipes that require
imputed model inputs without a categorical, leakage-safe fit step.

[PR #230](https://github.com/NVIDIA/structured-data-models/pull/230/files)
shows the intended composition point: categorical features are routed through
`ToNumerical` before the shared numerical pipeline. The categorical route can
instead be written as:

```python
StypeDispatch(
    categorical=[
        CategoricalImpute(strategy="most_frequent"),
        ToNumerical(),
    ],
)
```

The recipe fits this route on context or training rows and reuses its fitted
state for query rows. Query values never influence the selected fill category.
A transform input must use the fitted categorical column names, column order,
and category vocabularies in the same order. A mismatch raises instead of
silently applying a fitted code to a different value. Independently tensorized
tables must therefore share their encoded schema before using this processor.

## Decision

Add a dedicated categorical processor with one strategy:

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

The first strategy matches KumoRFM's `MOST_FREQUENT` behavior without assuming
that category vocabularies are already frequency-sorted. It also provides an
explicit opt-in before the `-1` sentinel from TabICL-style encoding crosses
into a numerical-only model pipeline.

## Why a separate processor

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
Value-based remapping across independently inferred vocabularies is also left
for a separate schema-alignment layer; it is broader than missing-value
imputation and may need to add values absent from the query vocabulary.

## Evidence

- [TabICL preprocessing](https://github.com/soda-inria/tabicl/blob/46b91961db4f8873dd049ec09990698a435e1e29/src/tabicl/_sklearn/preprocessing.py)
  encodes missing and unknown categorical values as `-1` while imputing only
  numerical columns.
- [KumoRFM categorical encoding](https://github.com/kumo-ai/kumo-ml/blob/6cb93eff313fbf770d33183098fe3ed2ec55accf/kumoml/common/nn/fused_encoder/fused_categorical.py#L146)
  maps missing values to the most frequent category for its
  `MOST_FREQUENT` strategy.
