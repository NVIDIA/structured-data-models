from typing import Literal

import pandas as pd
import pytest
import torch

from sdm import CategoricalTensor, EnsembleTable, StringTensor, TableTensor
from sdm.processing import AlignCategories, EnsembleProcessor
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
            code=torch.tensor(values, dtype=torch.int32, device=device),
            categories=tuple(
                StringTensor.from_list(category, device=device)
                for category in categories
            ),
        ),
    )


@withCUDA
def test_align_categories_remaps_independent_vocabularies(
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

    output = AlignCategories().fit(context).transform(query)

    assert torch.equal(
        output.categorical.code,
        torch.tensor(
            [[-1, 1], [1, -1], [0, 1], [-1, -1]],
            dtype=torch.int32,
            device=device,
        ),
    )
    assert output.categorical.categories[0].tolist() == ["red", "blue"]
    assert output.categorical.categories[1].tolist() == ["x", "y"]
    assert output.categorical.device == device


def test_align_categories_removes_query_only_joint_vocabulary() -> None:
    table = TableTensor.from_pandas(
        pd.DataFrame({"kind": ["red", "blue", "green", "blue"]}),
        stypes={"kind": "categorical"},
    )

    processor = AlignCategories().fit(table[:2])
    context = processor.transform(table[:2])
    query = processor.transform(table[2:])

    assert context.categorical.categories[0].tolist() == ["red", "blue"]
    assert context.categorical.code.squeeze(-1).tolist() == [0, 1]
    assert query.categorical.categories[0].tolist() == ["red", "blue"]
    assert query.categorical.code.squeeze(-1).tolist() == [-1, 1]


@pytest.mark.parametrize("sort_by", ["code", "frequency", "value"])
def test_align_categories_orders_joint_vocabulary(
    sort_by: Literal["code", "frequency", "value"],
) -> None:
    table = TableTensor.from_pandas(
        pd.DataFrame({"kind": ["blue", "red", "red", "red"]}),
        stypes={"kind": "categorical"},
    )

    processor = AlignCategories(sort_by).fit(table[:3])
    context = processor.transform(table[:3])
    query = processor.transform(table[3:])

    if sort_by == "code":
        assert context.categorical.categories[0].tolist() == ["blue", "red"]
        assert context.categorical.code.squeeze(-1).tolist() == [
            0,
            1,
            1,
        ]
        assert query.categorical.code.squeeze(-1).tolist() == [1]
    elif sort_by == "frequency":
        assert context.categorical.categories[0].tolist() == ["red", "blue"]
        assert context.categorical.code.squeeze(-1).tolist() == [
            1,
            0,
            0,
        ]
        assert query.categorical.code.squeeze(-1).tolist() == [0]
    else:
        assert sort_by == "value"
        assert context.categorical.categories[0].tolist() == ["blue", "red"]
        assert context.categorical.code.squeeze(-1).tolist() == [
            0,
            1,
            1,
        ]
        assert query.categorical.code.squeeze(-1).tolist() == [1]


@withCUDA
def test_align_categories_orders_values(
    device: torch.device,
) -> None:
    context = _table(
        [[0], [1], [2], [-1]],
        columns=("kind",),
        categories=(("zebra", "éclair", "ant", "unused"),),
        device=device,
    )
    query = _table(
        [[0], [1], [2], [-1]],
        columns=("kind",),
        categories=(("éclair", "zebra", "ant"),),
        device=device,
    )

    processor = AlignCategories(sort_by="value")
    context_output = processor.fit_transform(context)
    query_output = processor.transform(query)

    assert context_output.categorical.categories[0].tolist() == [
        "ant",
        "zebra",
        "éclair",
    ]
    assert torch.equal(
        context_output.categorical.code,
        torch.tensor([[1], [2], [0], [-1]], dtype=torch.int32, device=device),
    )
    assert torch.equal(
        query_output.categorical.code,
        torch.tensor([[2], [1], [0], [-1]], dtype=torch.int32, device=device),
    )


@withCUDA
def test_align_categories_numeric_values(device: torch.device) -> None:
    context = TableTensor(
        columns={"categorical": ("value",)},
        categorical=CategoricalTensor(
            code=torch.tensor(
                [[0], [1], [0]],
                dtype=torch.int32,
                device=device,
            ),
            categories=(torch.tensor([20, 10, 30], device=device),),
        ),
    )
    query = TableTensor(
        columns={"categorical": ("value",)},
        categorical=CategoricalTensor(
            code=torch.tensor(
                [[0], [1], [2], [-1]],
                dtype=torch.int32,
                device=device,
            ),
            categories=(torch.tensor([10, 40, 20], device=device),),
        ),
    )

    output = AlignCategories().fit(context).transform(query)

    assert torch.equal(
        output.categorical.code,
        torch.tensor(
            [[1], [-1], [0], [-1]],
            dtype=torch.int32,
            device=device,
        ),
    )
    assert torch.equal(
        output.categorical.categories[0],
        torch.tensor([20, 10], device=device),
    )


@withCUDA
def test_align_categories_does_not_match_nan_category_values(
    device: torch.device,
) -> None:
    context = TableTensor(
        columns={"categorical": ("value",)},
        categorical=CategoricalTensor(
            code=torch.tensor([[0], [1]], dtype=torch.int32, device=device),
            categories=(torch.tensor([torch.nan, 1.0], device=device),),
        ),
    )
    query = TableTensor(
        columns={"categorical": ("value",)},
        categorical=CategoricalTensor(
            code=torch.tensor(
                [[0], [1], [2]],
                dtype=torch.int32,
                device=device,
            ),
            categories=(torch.tensor([torch.nan, 2.0, 1.0], device=device),),
        ),
    )

    output = AlignCategories().fit(context).transform(query)

    assert output.categorical.code.squeeze(-1).tolist() == [-1, -1, -1]


def test_align_categories_scales_to_large_numeric_vocabulary() -> None:
    size = 10_000
    codes = torch.arange(size, dtype=torch.int32).unsqueeze(-1)
    context = TableTensor(
        columns={"categorical": ("value",)},
        categorical=CategoricalTensor(
            code=codes,
            categories=(torch.arange(size),),
        ),
    )
    query = TableTensor(
        columns={"categorical": ("value",)},
        categorical=CategoricalTensor(
            code=codes,
            categories=(torch.arange(size).flip(0),),
        ),
    )

    output = AlignCategories().fit(context).transform(query)

    assert torch.equal(
        output.categorical.code.squeeze(-1),
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
def test_align_categories_unsigned_pandas_values(
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

    output = AlignCategories().fit(context).transform(query)

    assert output.categorical.code.squeeze(-1).tolist() == [1, -1, 0]
    assert output.categorical.categories[0].dtype == getattr(torch, dtype)
    assert output.categorical.categories[0].tolist() == [largest, 1]


@withCUDA
@pytest.mark.parametrize(
    ("dtype", "largest"),
    [
        ("uint16", 2**16 - 1),
        ("uint32", 2**32 - 1),
        ("uint64", 2**63 + 1),
    ],
)
def test_align_categories_orders_unsigned_pandas_values(
    dtype: str,
    largest: int,
    device: torch.device,
) -> None:
    context = TableTensor.from_pandas(
        pd.DataFrame(
            {"value": pd.Series([largest, 1], dtype=dtype)},
        ),
        stypes={"value": "categorical"},
        device=device,
    )
    query = TableTensor.from_pandas(
        pd.DataFrame(
            {"value": pd.Series([1, largest - 1, largest], dtype=dtype)},
        ),
        stypes={"value": "categorical"},
        device=device,
    )

    output = AlignCategories(sort_by="value").fit(context).transform(query)

    assert output.categorical.code.squeeze(-1).tolist() == [0, -1, 1]
    assert output.categorical.categories[0].dtype == getattr(torch, dtype)
    assert output.categorical.categories[0].tolist() == [1, largest]


def test_align_categories_all_missing_context_has_empty_vocabulary() -> None:
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

    output = AlignCategories().fit(context).transform(query)

    assert output.categorical.categories[0].numel() == 0
    assert output.categorical.code.squeeze(-1).tolist() == [-1, -1, -1]


def test_align_categories_all_missing_pandas_context_accepts_strings() -> None:
    context = TableTensor.from_pandas(
        pd.DataFrame({"kind": [None, None]}),
        stypes={"kind": "categorical"},
    )
    query = TableTensor.from_pandas(
        pd.DataFrame({"kind": ["red", None]}),
        stypes={"kind": "categorical"},
    )

    output = AlignCategories(sort_by="value").fit(context).transform(query)

    assert output.categorical.categories[0].numel() == 0
    assert output.categorical.code.squeeze(-1).tolist() == [-1, -1]


def test_align_categories_rejects_changed_category_value_type() -> None:
    processor = AlignCategories().fit(
        _table([[0]], columns=("kind",), categories=(("red",),))
    )
    query = TableTensor(
        columns={"categorical": ("kind",)},
        categorical=CategoricalTensor(
            code=torch.tensor([[0]], dtype=torch.int32),
            categories=(torch.tensor([1]),),
        ),
    )

    with pytest.raises(NotImplementedError):
        processor.transform(query)


def test_align_categories_is_an_ensemble_processor() -> None:
    assert issubclass(AlignCategories, EnsembleProcessor)


def test_align_categories_keeps_fitted_vocabularies_per_representation() -> (
    None
):
    first = _table(
        [[0], [1]],
        columns=("kind",),
        categories=(("red", "blue", "green"),),
    )
    second = _table(
        [[1], [2]],
        columns=("kind",),
        categories=(("red", "blue", "green"),),
    )
    processor = AlignCategories()
    context = EnsembleTable.from_representations(
        (first, second),
        member_representation_ids=(0, 1, 0, 1),
    )

    transformed = processor.fit_transform_ensemble(context)

    assert transformed.representation(0).categorical.categories[
        0
    ].tolist() == [
        "red",
        "blue",
    ]
    assert transformed.representation(1).categorical.categories[
        0
    ].tolist() == [
        "blue",
        "green",
    ]

    query = _table(
        [[0], [1], [2]],
        columns=("kind",),
        categories=(("red", "blue", "green"),),
    )
    query_output = processor.transform_ensemble(
        EnsembleTable.from_representations(
            (query, query),
            member_representation_ids=(0, 1, 0, 1),
        )
    )
    assert query_output.representation(0).categorical.code.tolist() == [
        [0],
        [1],
        [-1],
    ]
    assert query_output.representation(1).categorical.code.tolist() == [
        [-1],
        [0],
        [1],
    ]


def test_align_categories_keeps_shared_output_packed() -> None:
    table = _table(
        [[0], [1]],
        columns=("kind",),
        categories=(("red", "blue"),),
    )
    output = AlignCategories().fit_transform_ensemble(
        EnsembleTable(table, num_members=8)
    )

    assert (
        sum(packed.size(0) for packed in output.iter_packed_representations())
        == 1
    )
