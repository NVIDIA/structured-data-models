import pytest
import torch
from sdm import CategoricalTensor, StringTensor, TableTensor
from sdm.processing import Pipeline, Processor, Recipe, StandardScale
from sdm.processing.base import InvertibleMixin


def test_public_recipe_imports() -> None:
    from sdm.processing.pipeline import Pipeline as CanonicalPipeline
    from sdm.processing.recipe import Recipe as CanonicalRecipe

    assert Pipeline is CanonicalPipeline
    assert Recipe is CanonicalRecipe


class Add(Processor):
    requires_fit = False

    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = value

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input + self.value


class Scale(Processor, InvertibleMixin):
    requires_fit = False

    def __init__(self, factor: float) -> None:
        super().__init__()
        self.factor = factor

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input * self.factor

    def _inverse_transform(self, input: torch.Tensor) -> torch.Tensor:
        return input / self.factor


class Shift(Processor, InvertibleMixin):
    requires_fit = False

    def __init__(self, offset: float) -> None:
        super().__init__()
        self.offset = offset

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input + self.offset

    def _inverse_transform(self, input: torch.Tensor) -> torch.Tensor:
        return input - self.offset


class Identity(Processor):
    requires_fit = False

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input


class FailingProcessor(Processor):
    requires_fit = False

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("boom")


def _table(numerical: torch.Tensor | None = None) -> TableTensor:
    numerical = (
        torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        if numerical is None
        else numerical
    )
    categorical = CategoricalTensor(
        data=torch.tensor([[0], [1]], dtype=torch.int64),
        categories=(StringTensor.from_list(["a", "b"]),),
    )
    return TableTensor(
        columns={
            "numerical": ("x0", "x1"),
            "categorical": ("kind",),
        },
        numerical=numerical,
        categorical=categorical,
    )


def test_empty_pipeline_returns_input_table() -> None:
    table = _table()

    assert Pipeline().transform(table) is table
    assert Pipeline().fit_transform(table) is table
    assert Pipeline().inverse_transform(table) is table


def test_pipeline_preserves_declared_stage_order() -> None:
    table = _table()

    output = Pipeline([Add(1), Scale(2)]).transform(table)
    reversed_output = Pipeline([Scale(2), Add(1)]).transform(table)

    assert torch.equal(output.numerical, (table.numerical + 1) * 2)
    assert not torch.equal(output.numerical, reversed_output.numerical)


def test_pipeline_rejects_non_stage() -> None:
    with pytest.raises(TypeError, match="Expected a Processor step"):
        Pipeline([object()])  # ty: ignore[invalid-argument-type]


def test_recipe_normalizes_empty_phases_and_describes() -> None:
    recipe = Recipe(features=[Add(1)], target=None, output=[])

    assert len(recipe.features) == 1
    assert len(recipe.target) == 0
    assert len(recipe.output) == 0
    assert "features: Add" in repr(recipe)
    assert "target: identity" in repr(recipe)


def test_preprocess_transforms_numerical_and_passes_categorical() -> None:
    table = _table()

    output = Pipeline([Add(5)]).transform(table)

    assert torch.equal(output.numerical, table.numerical + 5)
    assert output.categorical is table.categorical
    assert output.columns == table.columns


def test_pipeline_returns_input_when_numerical_is_unchanged() -> None:
    table = _table()

    output = Pipeline([Identity(), Identity()]).transform(table)

    assert output is table


def test_recipe_execution_order_around_stub_model() -> None:
    order: list[str] = []

    class Recorder(Processor, InvertibleMixin):
        requires_fit = False

        def __init__(self, name: str) -> None:
            super().__init__()
            self.name = name

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            order.append(self.name)
            return input

        def _inverse_transform(self, input: torch.Tensor) -> torch.Tensor:
            order.append(f"{self.name}-inverse")
            return input

    recipe = Recipe(
        features=[Recorder("features")],
        target=[Recorder("target")],
        output=[Recorder("output")],
    )
    table = _table()

    target_input = recipe.fit_transform_target(table)
    model_input = recipe.transform_features(table)
    order.append("model")
    target_output = recipe.inverse_transform_target(model_input)
    recipe.transform_output(target_output)

    assert target_input is not None
    assert order == [
        "target",
        "features",
        "model",
        "target-inverse",
        "output",
    ]


def test_recipe_runtime_error_includes_phase_and_step_position() -> None:
    recipe = Recipe(features=[Add(1), FailingProcessor(), Add(2)])

    with pytest.raises(RuntimeError, match=r"features step 1"):
        recipe.transform_features(_table())


def test_recipe_bad_step_output_includes_phase_and_step_position() -> None:
    class BadOutput(Processor):
        requires_fit = False

        def forward(self, input: torch.Tensor) -> object:  # ty: ignore[invalid-method-override]
            return object()

    recipe = Recipe(features=[Add(1), BadOutput()])

    with pytest.raises(TypeError, match=r"features step 1"):
        recipe.transform_features(_table())


def test_recipe_non_invertible_target_error_has_context() -> None:
    recipe = Recipe(target=[Add(1)])

    with pytest.raises(TypeError, match=r"target step 0"):
        recipe.inverse_transform_target(_table())


def test_target_forward_then_inverse_round_trips() -> None:
    recipe = Recipe(target=[Scale(2), Shift(1)])
    table = _table()

    transformed = recipe.fit_transform_target(table)
    restored = recipe.inverse_transform_target(transformed)

    assert torch.equal(transformed.numerical, table.numerical * 2 + 1)
    assert torch.allclose(restored.numerical, table.numerical)


def test_fit_transform_returns_features_and_target() -> None:
    recipe = Recipe(features=[Add(5)], target=[Scale(2)])
    features = _table()
    target = _table(torch.tensor([[10.0, 20.0], [30.0, 40.0]]))

    out_features, out_target = recipe.fit_transform(features, target)

    assert torch.equal(out_features.numerical, features.numerical + 5)
    assert torch.equal(out_target.numerical, target.numerical * 2)


def test_fit_transform_accepts_and_returns_tabletensor() -> None:
    table = _table()

    output = Pipeline([StandardScale()]).fit_transform(table)

    assert isinstance(output, TableTensor)
    assert output.categorical is table.categorical


def test_inverse_transform_runs_stages_in_reverse_order() -> None:
    table = _table(torch.tensor([[10.0, 20.0], [30.0, 40.0]]))

    output = Pipeline([Shift(1), Scale(2)]).inverse_transform(table)

    assert torch.equal(output.numerical, table.numerical / 2 - 1)
