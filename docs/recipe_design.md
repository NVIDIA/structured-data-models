# Recipe 

Recipes provide an inspectable processing contract around a model.
A `Recipe` owns three ordered slots (`preprocess`, `target`, and
`postprocess`). Configure each slot with a list of processing stages. Each
slot accepts and returns a `TableTensor`.

```python
from sdm.processing import Clip, Recipe, StandardScale

recipe = Recipe(
    preprocess=[
        Clip(q_low=0.01, q_high=0.99),
        StandardScale(),
    ],
    target=[StandardScale()],
    postprocess=[],
)
```

Fit and apply the preprocessing slot before calling the model:

```python
training_table = recipe.fit_transform_preprocess(train_data)
model_input = recipe.transform_preprocess(test_data)

prediction = model(model_input)
prediction = recipe.transform_postprocess(prediction)
```

Each slot applies stages to the numerical block and rebuilds the
`TableTensor` with categorical blocks unchanged.
