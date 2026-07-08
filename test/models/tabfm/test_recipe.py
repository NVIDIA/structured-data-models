from typing import cast

import torch
from sdm import CategoricalTensor, TableTensor
from sdm.models import TabFM
from sdm.models.tabfm import (
    default_classification_recipe,
    default_regression_recipe,
)
from sdm.processing import (
    Clamp,
    Identity,
    MeanImpute,
    Sequential,
    SigmaClip,
    StandardScale,
)


def _numerical_table(data: torch.Tensor) -> TableTensor:
    return TableTensor(
        columns={
            "numerical": [f"feature_{index}" for index in range(data.size(1))]
        },
        numerical=data,
    )


def test_tabfm_recipe_uses_upstream_numeric_order() -> None:
    classification = default_classification_recipe()
    regression = default_regression_recipe()

    for recipe in (classification, regression):
        features = cast(Sequential, recipe.features)
        assert [type(step) for step in features.steps] == [
            MeanImpute,
            StandardScale,
            Clamp,
            SigmaClip,
        ]
        scaler = features.steps[1]
        clamp = features.steps[2]
        sigma_clip = features.steps[3]
        assert isinstance(scaler, StandardScale)
        assert scaler.epsilon == 1e-6
        assert isinstance(clamp, Clamp)
        assert clamp.min_value == -100.0
        assert clamp.max_value == 100.0
        assert isinstance(sigma_clip, SigmaClip)
        assert sigma_clip.threshold == 4.0

    classification_target = cast(Sequential, classification.target)
    regression_target = cast(Sequential, regression.target)
    assert isinstance(classification_target.steps[0], Identity)
    assert isinstance(regression_target.steps[0], StandardScale)


def test_tabfm_recipe_fits_feature_state_on_context_only() -> None:
    context = _numerical_table(
        torch.tensor(
            [
                [0.0, float("nan")],
                [2.0, 4.0],
                [4.0, 8.0],
            ]
        )
    )
    query = _numerical_table(torch.tensor([[1000.0, -1000.0]]))
    recipe = default_regression_recipe()
    features = cast(Sequential, recipe.features)

    context_output = features.fit_transform(context)
    query_output = features.transform(query)

    imputer, scaler, clamp, sigma_clip = features.steps
    assert isinstance(imputer, MeanImpute)
    assert isinstance(scaler, StandardScale)
    assert isinstance(clamp, Clamp)
    assert isinstance(sigma_clip, SigmaClip)
    torch.testing.assert_close(imputer._mean, torch.tensor([2.0, 6.0]))
    torch.testing.assert_close(scaler.mean, torch.tensor([2.0, 6.0]))
    expected_scale = torch.tensor([8.0 / 3.0, 8.0 / 3.0]).sqrt() + 1e-6
    torch.testing.assert_close(scaler.scale, expected_scale)
    assert torch.isfinite(context_output.numerical).all()
    assert torch.isfinite(query_output.numerical).all()

    leaked_recipe = default_regression_recipe()
    leaked_features = cast(Sequential, leaked_recipe.features)
    combined = cast(TableTensor, torch.cat([context, query], dim=0))
    leaked_features.fit(combined)
    leaked_scaler = leaked_features.steps[1]
    assert isinstance(leaked_scaler, StandardScale)
    assert not torch.equal(leaked_scaler.mean, scaler.mean)


def test_tabfm_regression_target_round_trip() -> None:
    recipe = default_regression_recipe()
    target_processor = cast(Sequential, recipe.target)
    target = _numerical_table(torch.tensor([[10.0], [20.0], [40.0], [80.0]]))
    prediction = _numerical_table(torch.tensor([[-1.0], [0.0], [1.0]]))

    transformed = target_processor.fit_transform(target)
    restored = target_processor.inverse_transform(transformed)
    restored_prediction = target_processor.inverse_transform(prediction)

    torch.testing.assert_close(restored.numerical, target.numerical)
    scaler = target_processor.steps[0]
    assert isinstance(scaler, StandardScale)
    torch.testing.assert_close(
        restored_prediction.numerical,
        prediction.numerical * scaler.scale + scaler.mean,
    )


def test_tabfm_classification_target_is_unchanged() -> None:
    recipe = default_classification_recipe()
    target_processor = cast(Sequential, recipe.target)
    target = TableTensor(
        columns={"categorical": ["target"]},
        categorical=CategoricalTensor(
            data=torch.tensor([[0], [2], [1], [2]], dtype=torch.int32),
            categories=(torch.arange(3),),
        ),
    )

    transformed = target_processor.fit_transform(target)
    restored = target_processor.inverse_transform(transformed)

    assert transformed is target
    assert restored is target


def test_tabfm_recipe_keeps_categoricals_outside_numeric_pipeline() -> None:
    numerical = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    categorical = CategoricalTensor(
        data=torch.tensor([[0], [1], [0], [2]], dtype=torch.int32),
        categories=(torch.arange(3),),
    )
    table = TableTensor(
        columns={"numerical": ["value"], "categorical": ["kind"]},
        numerical=numerical,
        categorical=categorical,
    )
    recipe = default_classification_recipe()
    features = cast(Sequential, recipe.features)
    original_categorical = table.categorical.as_tensor().clone()

    numerical_table = TableTensor(
        columns={"numerical": ["value"]},
        numerical=table.numerical,
    )
    transformed_numerical = features.fit_transform(numerical_table)

    assert transformed_numerical.shape == numerical_table.shape
    torch.testing.assert_close(
        table.categorical.as_tensor(),
        original_categorical,
    )


def test_tabfm_model_exposes_regression_recipe() -> None:
    recipe = TabFM(pretrained=False).default_recipe()

    assert isinstance(recipe.features, Sequential)
    assert isinstance(recipe.target, Sequential)
    assert isinstance(recipe.target.steps[0], StandardScale)
