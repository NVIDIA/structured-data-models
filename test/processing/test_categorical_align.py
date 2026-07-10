import pandas as pd
import pytest
import torch
from sdm import CategoricalTensor, StringTensor, TableTensor
from sdm.processing import CategoricalAlign
from sdm.testing import withCUDA


def _table(
    values: list[list[int]],
    *,
    categories: tuple[tuple[str, ...], ...],
    columns: tuple[str, ...] = ("kind", "segment"),
    device: torch.device | str | None = None,
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
def test_categorical_align_remaps_independent_vocabularies(
    device: torch.device,
) -> None:
    context = _table(
        [[0, 0], [1, 1], [0, -1]],
        categories=(("red", "blue", "unused"), ("x", "y")),
        device=device,
    )
    query = _table(
        [[0, 0], [1, 1], [2, 0], [-1, -1]],
        categories=(("green", "blue", "red"), ("y", "z")),
        device=device,
    )

    output = CategoricalAlign().fit(context).transform(query)

    assert torch.equal(
        output.categorical.as_tensor(),
        torch.tensor(
            [[-1, 1], [1, -1], [0, 1], [-1, -1]],
            dtype=torch.int32,
            device=device,
        ),
    )
    assert output.categorical.categories[0].tolist() == ["red", "blue"]
    assert output.categorical.categories[1].tolist() == ["x", "y"]
    assert output.categorical.device == device


def test_categorical_align_removes_query_only_joint_vocabulary() -> None:
    table = TableTensor.from_pandas(
        pd.DataFrame({"kind": ["red", "blue", "green", "blue"]}),
        stypes={"kind": "categorical"},
    )

    processor = CategoricalAlign().fit(table[:2])
    context = processor.transform(table[:2])
    query = processor.transform(table[2:])

    assert context.categorical.categories[0].tolist() == ["red", "blue"]
    assert context.categorical.as_tensor().squeeze(-1).tolist() == [0, 1]
    assert query.categorical.categories[0].tolist() == ["red", "blue"]
    assert query.categorical.as_tensor().squeeze(-1).tolist() == [-1, 1]


def test_categorical_align_joint_vocabulary_uses_context_order() -> None:
    table = TableTensor.from_pandas(
        pd.DataFrame({"kind": ["blue", "red", "blue"]}),
        stypes={"kind": "categorical"},
    )

    processor = CategoricalAlign().fit(table[1:])
    context = processor.transform(table[1:])
    query = processor.transform(table[:1])

    assert context.categorical.categories[0].tolist() == ["red", "blue"]
    assert context.categorical.as_tensor().squeeze(-1).tolist() == [0, 1]
    assert query.categorical.as_tensor().squeeze(-1).tolist() == [1]


def test_categorical_align_numeric_values() -> None:
    context = TableTensor(
        columns={"categorical": ("value",)},
        categorical=CategoricalTensor(
            data=torch.tensor([[0], [1], [0]], dtype=torch.int32),
            categories=(torch.tensor([20, 10, 30]),),
        ),
    )
    query = TableTensor(
        columns={"categorical": ("value",)},
        categorical=CategoricalTensor(
            data=torch.tensor([[0], [1], [2], [-1]], dtype=torch.int32),
            categories=(torch.tensor([10, 40, 20]),),
        ),
    )

    output = CategoricalAlign().fit(context).transform(query)

    assert torch.equal(
        output.categorical.as_tensor(),
        torch.tensor([[1], [-1], [0], [-1]], dtype=torch.int32),
    )
    assert torch.equal(
        output.categorical.categories[0],
        torch.tensor([20, 10]),
    )


def test_categorical_align_matches_nan_category_values() -> None:
    context = TableTensor(
        columns={"categorical": ("value",)},
        categorical=CategoricalTensor(
            data=torch.tensor([[0], [1]], dtype=torch.int32),
            categories=(torch.tensor([torch.nan, 1.0]),),
        ),
    )
    query = TableTensor(
        columns={"categorical": ("value",)},
        categorical=CategoricalTensor(
            data=torch.tensor([[0], [1], [2]], dtype=torch.int32),
            categories=(torch.tensor([torch.nan, 2.0, 1.0]),),
        ),
    )

    output = CategoricalAlign().fit(context).transform(query)

    assert output.categorical.as_tensor().squeeze(-1).tolist() == [0, -1, 1]


def test_categorical_align_scales_to_large_numeric_vocabulary() -> None:
    size = 10_000
    codes = torch.arange(size, dtype=torch.int32).unsqueeze(-1)
    context = TableTensor(
        columns={"categorical": ("value",)},
        categorical=CategoricalTensor(
            data=codes,
            categories=(torch.arange(size),),
        ),
    )
    query = TableTensor(
        columns={"categorical": ("value",)},
        categorical=CategoricalTensor(
            data=codes,
            categories=(torch.arange(size).flip(0),),
        ),
    )

    output = CategoricalAlign().fit(context).transform(query)

    assert torch.equal(
        output.categorical.as_tensor().squeeze(-1),
        torch.arange(size - 1, -1, -1, dtype=torch.int32),
    )


@pytest.mark.parametrize(
    ("dtype", "largest"),
    [
        ("uint16", 2**16 - 1),
        ("uint32", 2**32 - 1),
        ("uint64", 2**63 + 1),
    ],
)
def test_categorical_align_unsigned_pandas_values(
    dtype: str,
    largest: int,
) -> None:
    context = TableTensor.from_pandas(
        pd.DataFrame(
            {"value": pd.Series([largest, 1], dtype=dtype)},
        ),
        stypes={"value": "categorical"},
    )
    query = TableTensor.from_pandas(
        pd.DataFrame(
            {"value": pd.Series([1, largest - 1, largest], dtype=dtype)},
        ),
        stypes={"value": "categorical"},
    )

    output = CategoricalAlign().fit(context).transform(query)

    assert output.categorical.as_tensor().squeeze(-1).tolist() == [1, -1, 0]
    assert output.categorical.categories[0].dtype == getattr(torch, dtype)
    assert output.categorical.categories[0].tolist() == [largest, 1]


def test_categorical_align_all_missing_context_has_empty_vocabulary() -> None:
    context = _table(
        [[-1], [-1]],
        columns=("kind",),
        categories=(("red", "blue"),),
    )
    query = _table(
        [[0], [1], [-1]],
        columns=("kind",),
        categories=(("red", "green"),),
    )

    output = CategoricalAlign().fit(context).transform(query)

    assert output.categorical.categories[0].numel() == 0
    assert output.categorical.as_tensor().squeeze(-1).tolist() == [-1, -1, -1]


def test_categorical_align_all_missing_pandas_context_accepts_strings() -> (
    None
):
    context = TableTensor.from_pandas(
        pd.DataFrame({"kind": [None, None]}),
        stypes={"kind": "categorical"},
    )
    query = TableTensor.from_pandas(
        pd.DataFrame({"kind": ["red", None]}),
        stypes={"kind": "categorical"},
    )

    output = CategoricalAlign().fit(context).transform(query)

    assert output.categorical.categories[0].numel() == 0
    assert output.categorical.as_tensor().squeeze(-1).tolist() == [-1, -1]


def test_categorical_align_rejects_changed_columns() -> None:
    processor = CategoricalAlign().fit(
        _table([[0]], columns=("kind",), categories=(("red",),))
    )
    query = _table(
        [[0]],
        columns=("segment",),
        categories=(("red",),),
    )

    with pytest.raises(ValueError, match=r"columns.*fitted names and order"):
        processor.transform(query)


def test_categorical_align_rejects_changed_category_value_type() -> None:
    processor = CategoricalAlign().fit(
        _table([[0]], columns=("kind",), categories=(("red",),))
    )
    query = TableTensor(
        columns={"categorical": ("kind",)},
        categorical=CategoricalTensor(
            data=torch.tensor([[0]], dtype=torch.int32),
            categories=(torch.tensor([1]),),
        ),
    )

    with pytest.raises(ValueError, match=r"value types.*kind"):
        processor.transform(query)


def test_categorical_align_rejects_lossy_numeric_dtype_change() -> None:
    context = TableTensor(
        columns={"categorical": ("value",)},
        categorical=CategoricalTensor(
            data=torch.tensor([[0]], dtype=torch.int32),
            categories=(torch.tensor([16_777_217], dtype=torch.int64),),
        ),
    )
    query = TableTensor(
        columns={"categorical": ("value",)},
        categorical=CategoricalTensor(
            data=torch.tensor([[0]], dtype=torch.int32),
            categories=(torch.tensor([16_777_216], dtype=torch.float32),),
        ),
    )

    with pytest.raises(ValueError, match=r"value dtypes.*value"):
        CategoricalAlign().fit(context).transform(query)


def test_categorical_align_rejects_complex_category_values() -> None:
    context = TableTensor(
        columns={"categorical": ("value",)},
        categorical=CategoricalTensor(
            data=torch.tensor([[0]], dtype=torch.int32),
            categories=(torch.tensor([1 + 2j]),),
        ),
    )

    with pytest.raises(ValueError, match=r"complex.*value"):
        CategoricalAlign().fit_transform(context)


@pytest.mark.parametrize("during_fit", [True, False])
def test_categorical_align_rejects_out_of_range_codes(
    during_fit: bool,
) -> None:
    invalid = _table(
        [[1]],
        columns=("kind",),
        categories=(("red",),),
    )
    if during_fit:
        with pytest.raises(ValueError, match=r"kind.*outside.*vocabulary"):
            CategoricalAlign().fit(invalid)
        return

    processor = CategoricalAlign().fit(
        _table([[0]], columns=("kind",), categories=(("red",),))
    )
    with pytest.raises(ValueError, match=r"kind.*outside.*vocabulary"):
        processor.transform(invalid)
