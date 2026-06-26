# Recipe Design

Recipes provide an inspectable processing contract around an external model.
V0 keeps the surface intentionally small: a `Recipe` owns three ordered
`Pipeline` slots (`preprocess`, `target`, and `postprocess`) and each slot
accepts and returns a `TableTensor`.

```python
from sdm.processing import Pipeline, Recipe, StandardScale

recipe = Recipe(
    preprocess=Pipeline([StandardScale()]),
    target=Pipeline([StandardScale()]),
    postprocess=Pipeline(),
)
```

The pipeline applies stages to the numerical block and rebuilds the
`TableTensor` with categorical blocks unchanged. Model invocation, sampling,
augmentation, semantic-type conversion, and label decoding are outside the V0
recipe contract.
