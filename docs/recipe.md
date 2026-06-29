# Recipe

Recipes provide an inspectable processing contract around a model.
A `Recipe` owns three ordered phases (`preprocess`, `target`, and
`postprocess`). Configure each phase with a list of processing steps. Each
phase accepts and returns a `TableTensor`.

```python
from sdm.processing import Clip, Recipe, StandardScale

preprocess_scale = StandardScale()
preprocess_scale.fit(train_data.numerical)

recipe = Recipe(
    preprocess=[
        Clip(q_low=0.01, q_high=0.99),
        preprocess_scale,
    ],
    target=[],
    postprocess=[],
)
```

Fit and apply the preprocessing phase before calling the model:

```python
training_table = recipe.fit_transform_preprocess(train_data)
model_input = recipe.transform_preprocess(test_data)

prediction = model(model_input)
prediction = recipe.transform_postprocess(prediction)
```

The `target` phase holds pre-fitted or stateless processors whose
`inverse_transform` converts model outputs back to the original space.
`Recipe` does not provide a `fit_target` method; fit target-side processors
before passing them to the recipe.

Each phase applies steps to the numerical block and rebuilds the
`TableTensor` with categorical blocks unchanged.
