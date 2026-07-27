import pandas as pd
import pytest
import torch
from sdm import CategoricalTensor, StringTensor, TableTensor
from sdm.processing import AlignCategories, SortCategories
from sdm.testing import withCUDA


def _string_table(
    values: list[list[int]],
    *,
    categories: tuple[str, ...],
    device: torch.device | str | None = None,
) -> TableTensor:
    return TableTensor(
        columns={"categorical": ("kind",)},
        categorical=CategoricalTensor(
            data=torch.tensor(values, dtype=torch.int32, device=device),
            categories=(StringTensor.from_list(categories, device=device),),
        ),
    )


@withCUDA
def test_sort_categories_orders_string_vocabulary_and_preserves_missing(
    device: torch.device,
) -> None:
    table = _string_table(
        [[0], [1], [2], [-1]],
        categories=("zebra", "ant", "bee"),
        device=device,
    )

    output = SortCategories().transform(table)

    assert output.categorical.categories[0].tolist() == ["ant", "bee", "zebra"]
    assert output.categorical.as_tensor().squeeze(-1).tolist() == [2, 0, 1, -1]
    assert output.categorical.device == device


def test_sort_categories_preserves_empty_vocabulary() -> None:
    table = _string_table([[-1], [-1]], categories=())

    output = SortCategories().transform(table)

    assert output.categorical.categories[0].numel() == 0
    assert output.categorical.as_tensor().squeeze(-1).tolist() == [-1, -1]


@withCUDA
def test_sort_categories_orders_numeric_vocabulary_on_table_device(
    device: torch.device,
) -> None:
    table = TableTensor(
        columns={"categorical": ("value",)},
        categorical=CategoricalTensor(
            data=torch.tensor([[0], [1], [2], [-1]], dtype=torch.int32, device=device),
            categories=(torch.tensor([20, 10, 30], device=device),),
        ),
    )

    output = SortCategories().transform(table)

    torch.testing.assert_close(
        output.categorical.categories[0],
        torch.tensor([10, 20, 30], device=device),
    )
    assert output.categorical.as_tensor().squeeze(-1).tolist() == [1, 0, 2, -1]
    assert output.categorical.device == device


@withCUDA
def test_sort_categories_orders_nan_after_finite_numeric_values(
    device: torch.device,
) -> None:
    table = TableTensor(
        columns={"categorical": ("value",)},
        categorical=CategoricalTensor(
            data=torch.tensor([[0], [1], [-1]], dtype=torch.int32, device=device),
            categories=(torch.tensor([torch.nan, 1.0], device=device),),
        ),
    )

    output = SortCategories().transform(table)

    torch.testing.assert_close(
        output.categorical.categories[0],
        torch.tensor([1.0, torch.nan], device=device),
        equal_nan=True,
    )
    assert output.categorical.as_tensor().squeeze(-1).tolist() == [1, 0, -1]


@pytest.mark.parametrize(
    ("dtype", "largest"),
    [
        ("uint16", 2**16 - 1),
        ("uint32", 2**32 - 1),
        ("uint64", 2**63 + 1),
    ],
)
def test_sort_categories_orders_unsigned_pandas_vocabulary(
    dtype: str,
    largest: int,
) -> None:
    table = TableTensor.from_pandas(
        pd.DataFrame(
            {"value": pd.Series([largest, 1, largest - 1], dtype=dtype)},
        ),
        stypes={"value": "categorical"},
    )

    output = SortCategories().transform(table)

    assert output.categorical.categories[0].dtype == getattr(torch, dtype)
    assert output.categorical.categories[0].tolist() == [1, largest - 1, largest]
    assert output.categorical.as_tensor().squeeze(-1).tolist() == [2, 0, 1]


def test_sort_categories_after_alignment_keeps_context_vocabulary() -> None:
    context = _string_table(
        [[0], [1], [0], [-1]],
        categories=("zebra", "ant", "unused"),
    )
    query = _string_table(
        [[0], [1], [2], [-1]],
        categories=("ant", "zebra", "bee"),
    )

    align = AlignCategories().fit(context)
    output = SortCategories().transform(align.transform(query))

    assert output.categorical.categories[0].tolist() == ["ant", "zebra"]
    assert output.categorical.as_tensor().squeeze(-1).tolist() == [0, 1, -1, -1]
