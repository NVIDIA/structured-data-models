import warnings

import pytest
import torch
from sdm import CategoricalTensor, StringTensor, Stype, TableTensor
from sdm.cache import Cache
from sdm.models.base import BaseModel
from sdm.processing import Recipe, ToNumerical


def _table() -> TableTensor:
    return TableTensor(
        columns={
            "numerical": ("age", "income"),
            "categorical": ("country", "segment"),
        },
        numerical=torch.tensor([[30.0, 100.0], [40.0, 200.0]]),
        categorical=CategoricalTensor(
            data=torch.tensor([[0, 1], [-1, 0]], dtype=torch.int64),
            categories=(
                StringTensor.from_list(["US", "DE"]),
                StringTensor.from_list(["small", "enterprise"]),
            ),
        ),
    )


def test_to_numerical_converts_categorical_stype_without_reencoding() -> None:
    table = _table()

    categorical_ids = table.categorical.as_tensor().to(table.numerical.dtype)

    output = ToNumerical().transform(table)

    assert isinstance(output, TableTensor)
    assert output.columns[Stype.numerical] == (
        "age",
        "income",
        "country",
        "segment",
    )
    assert output.columns[Stype.categorical] == ()
    assert torch.equal(output.numerical[..., :2], table.numerical)
    assert torch.equal(output.numerical[..., 2:], categorical_ids)
    assert output.categorical.size(-1) == 0


def test_to_numerical_respects_requested_dtype() -> None:
    output = ToNumerical(dtype=torch.float64).transform(_table())

    assert isinstance(output, TableTensor)
    assert output.numerical.dtype == torch.float64
    assert torch.equal(
        output.numerical,
        torch.tensor(
            [[30.0, 100.0, 0.0, 1.0], [40.0, 200.0, -1.0, 0.0]],
            dtype=torch.float64,
        ),
    )


def test_to_numerical_converts_categorical_only_table() -> None:
    table = TableTensor(
        columns={"categorical": ("country",)},
        categorical=CategoricalTensor(
            data=torch.tensor([[0], [-1]], dtype=torch.int64),
            categories=(StringTensor.from_list(["US"]),),
        ),
    )

    output = ToNumerical().transform(table)

    assert isinstance(output, TableTensor)
    assert output.columns[Stype.numerical] == ("country",)
    assert output.columns[Stype.categorical] == ()
    assert torch.equal(output.numerical, torch.tensor([[0.0], [-1.0]]))


def test_to_numerical_is_identity_for_already_numerical_table() -> None:
    table = TableTensor.from_tensor(
        torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        columns=("x0", "x1"),
    )

    assert ToNumerical().transform(table) is table


def test_to_numerical_requires_table_input() -> None:
    with pytest.raises(TypeError, match="TableTensor"):
        ToNumerical().transform(torch.ones(2, 3))


def test_to_numerical_keeps_source_category_vocabulary_available() -> None:
    table = _table()

    output = ToNumerical().transform(table)

    assert isinstance(output, TableTensor)
    assert table.categorical.categories[0].tolist() == ["US", "DE"]
    assert table.categorical.categories[1].tolist() == [
        "small",
        "enterprise",
    ]
    assert output.columns[Stype.categorical] == ()


def test_to_numerical_fits_recipe_feature_pipeline() -> None:
    table = _table()
    recipe = Recipe(features=[ToNumerical()])

    output = recipe.features.fit_transform(table)

    assert output.columns[Stype.numerical] == (
        "age",
        "income",
        "country",
        "segment",
    )
    assert output.size(-1) == output.numerical.size(-1)


class EchoModel(BaseModel):
    def _forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        *,
        cache: Cache | None = None,
    ) -> torch.Tensor:
        del y, cache
        return x


def test_to_numerical_prepares_features_for_base_model_boundary() -> None:
    table = _table()
    output = Recipe(features=[ToNumerical()]).features.transform(table)
    model = EchoModel()

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        model_output = model(output, torch.empty(0))

    assert len(captured) == 0
    assert model_output.shape == (2, 4)
