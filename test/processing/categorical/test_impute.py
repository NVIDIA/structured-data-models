from typing import Any, cast

import pytest
import torch
from sdm import CategoricalTensor, StringTensor, Stype, TableTensor
from sdm.processing import DispatchByStype, ImputeCategories, ToNumerical
from sdm.testing import onlyCUDA, withCUDA


def _table(
    values: list[list[int]],
    *,
    columns: tuple[str, ...] = ("kind", "segment"),
    categories: tuple[tuple[str, ...], ...] = (
        ("a", "b", "c"),
        ("x", "y"),
    ),
    device: torch.device | None = None,
) -> TableTensor:
    return TableTensor(
        columns={"categorical": columns},
        categorical=CategoricalTensor(
            data=torch.tensor(values, dtype=torch.int32, device=device),
            categories=tuple(
                StringTensor.from_list(category, device=device)
                for category in categories
            ),
        ),
    )


@withCUDA
def test_impute_categories_most_frequent(
    device: torch.device,
) -> None:
    context = _table(
        [[0, 1], [0, -1], [1, 1], [-1, 0]],
        device=device,
    )
    query = _table(
        [[-1, -1], [2, 0]],
        device=device,
    )
    processor = ImputeCategories().fit(context)

    output = processor.transform(query)

    assert torch.equal(
        processor._fill_values,
        torch.tensor([0, 1], device=device),
    )
    assert torch.equal(
        output.categorical.as_tensor(),
        torch.tensor([[0, 1], [2, 0]], dtype=torch.int32, device=device),
    )
    assert output.columns[Stype.categorical] == ("kind", "segment")
    for actual, expected in zip(
        output.categorical.categories,
        query.categorical.categories,
    ):
        assert torch.equal(actual, expected)


@onlyCUDA
def test_impute_categories_moves_fitted_processor_to_cuda() -> None:
    processor = ImputeCategories().fit(_table([[0, 1], [0, -1], [1, 0]]))
    processor = processor.to("cuda")
    query = _table([[-1, -1]], device=torch.device("cuda"))

    output = processor.transform(query)

    assert output.categorical.device.type == "cuda"
    assert torch.equal(
        output.categorical.as_tensor(),
        torch.tensor([[0, 0]], dtype=torch.int32, device="cuda"),
    )


@withCUDA
def test_impute_categories_tie_uses_lowest_code(
    device: torch.device,
) -> None:
    table = _table([[1, 0], [0, 1], [-1, -1]], device=device)

    output = ImputeCategories().fit_transform(table)

    assert torch.equal(
        output.categorical.as_tensor(),
        torch.tensor(
            [[1, 0], [0, 1], [0, 0]],
            dtype=torch.int32,
            device=device,
        ),
    )


def test_impute_categories_rejects_all_missing_column() -> None:
    table = _table([[0, -1], [1, -1]])

    with pytest.raises(ValueError, match=r"segment.*no observed values"):
        ImputeCategories().fit(table)


def test_impute_categories_rejects_unknown_strategy() -> None:
    with pytest.raises(ValueError, match="strategy must be 'most_frequent'"):
        ImputeCategories(strategy=cast(Any, "constant"))


@pytest.mark.parametrize(
    "categories",
    [
        (("b", "a", "c"), ("x", "y")),
        (("a", "b"), ("x", "y")),
    ],
)
def test_impute_categories_rejects_changed_vocabulary(
    categories: tuple[tuple[str, ...], ...],
) -> None:
    processor = ImputeCategories().fit(_table([[0, 0], [0, 1]]))
    query = _table([[-1, -1]], categories=categories)

    with pytest.raises(
        ValueError,
        match=r"vocabulary.*kind.*fitted values.*AlignCategories",
    ):
        processor.transform(query)


def test_impute_categories_rejects_reordered_columns() -> None:
    processor = ImputeCategories().fit(_table([[0, 0], [0, 1]]))
    query = _table(
        [[-1, -1]],
        columns=("segment", "kind"),
        categories=(("x", "y"), ("a", "b", "c")),
    )

    with pytest.raises(
        ValueError,
        match=r"vocabulary.*segment.*fitted values",
    ):
        processor.transform(query)


def test_impute_categories_rejects_out_of_range_code_during_fit() -> None:
    table = _table([[3, 0], [0, 1]])

    with pytest.raises(ValueError, match=r"kind.*outside.*vocabulary"):
        ImputeCategories().fit(table)


def test_impute_categories_rejects_out_of_range_code_during_transform() -> (
    None
):
    processor = ImputeCategories().fit(_table([[0, 0], [1, 1]]))
    query = _table([[3, -1]])

    with pytest.raises(ValueError, match=r"kind.*outside.*vocabulary"):
        processor.transform(query)


def test_impute_categories_composes_before_to_numerical() -> None:
    table = _table([[0, 0], [0, -1], [1, 1], [-1, 1]])
    processor = DispatchByStype(
        categorical=[ImputeCategories(), ToNumerical()],
    )

    output = processor.fit_transform(table)

    assert output.columns[Stype.numerical] == ("kind", "segment")
    assert output.columns[Stype.categorical] == ()
    assert torch.equal(
        output.numerical,
        torch.tensor(
            [[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 1.0]],
        ),
    )
