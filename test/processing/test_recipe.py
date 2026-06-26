import pytest
import torch
from sdm import CategoricalTensor, StringTensor, TableTensor
from sdm.processing import Pipeline, Processor, Recipe, StandardScale
from sdm.processing.base import InvertibleMixin, ensure_floating


def test_public_recipe_imports() -> None:
    from sdm.processing import Pipeline as PublicPipeline
    from sdm.processing import Recipe as PublicRecipe

    assert PublicPipeline is Pipeline
    assert PublicRecipe is Recipe


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


def test_ensure_floating_widens_integer_input() -> None:
    input = torch.tensor([[1, 2]], dtype=torch.int64)

    output = ensure_floating(input)

    assert output.dtype == torch.get_default_dtype()
    assert torch.equal(output, input.to(torch.get_default_dtype()))


def test_ensure_floating_rejects_unsupported_dtype() -> None:
    input = torch.tensor([[True, False]])

    with pytest.raises(TypeError, match="Expected a floating point"):
        ensure_floating(input)


def test_empty_pipeline_returns_input_table() -> None:
    table = _table()

    assert Pipeline().transform(table) is table
    assert Pipeline().fit_transform(table) is table
    assert Pipeline().inverse_transform(table) is table


def test_pipeline_preserves_declared_stage_order() -> None:
    table = _table()

    output = Pipeline([Add(1), Add(2)]).transform(table)

    assert torch.equal(output.numerical, table.numerical + 3)


def test_pipeline_rejects_non_stage() -> None:
    with pytest.raises(TypeError, match="Expected a Processor stage"):
        Pipeline([object()])  # type: ignore[list-item]


def test_recipe_normalizes_empty_slots_and_describes() -> None:
    recipe = Recipe(preprocess=[Add(1)], target=None, postprocess=[])

    assert len(recipe.preprocess) == 1
    assert len(recipe.target) == 0
    assert len(recipe.postprocess) == 0
    assert "preprocess: Add" in recipe.describe()
    assert "target: <empty>" in repr(recipe)


def test_preprocess_transforms_numerical_and_passes_categorical() -> None:
    table = _table()

    output = Pipeline([Add(5)]).transform(table)

    assert torch.equal(output.numerical, table.numerical + 5)
    assert output.categorical is table.categorical
    assert output.columns == table.columns


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
        preprocess=[Recorder("pre")],
        target=[Recorder("target")],
        postprocess=[Recorder("post")],
    )
    table = _table()

    model_input = recipe.transform_preprocess(table)
    order.append("model")
    target_output = recipe.inverse_transform_target(model_input)
    recipe.transform_postprocess(target_output)

    assert order == ["pre", "model", "target-inverse", "post"]


def test_recipe_runtime_error_includes_slot_and_stage_position() -> None:
    recipe = Recipe(preprocess=[Add(1), FailingProcessor(), Add(2)])

    with pytest.raises(RuntimeError, match=r"slot=preprocess, stage=1"):
        recipe.transform_preprocess(_table())


def test_fit_transform_accepts_tabletensor_with_integer_input() -> None:
    table = _table(torch.tensor([[1, 2], [3, 4]], dtype=torch.int64))

    output = Pipeline([StandardScale()]).fit_transform(table)

    assert isinstance(output, TableTensor)
    assert output.numerical.is_floating_point()
    assert output.categorical is table.categorical


def test_inverse_transform_runs_stages_in_reverse_order() -> None:
    table = _table(torch.tensor([[10.0, 20.0], [30.0, 40.0]]))

    output = Pipeline([Scale(2), Scale(5)]).inverse_transform(table)

    assert torch.equal(output.numerical, table.numerical / 10)
