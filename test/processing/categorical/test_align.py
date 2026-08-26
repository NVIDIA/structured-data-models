from typing import Literal

import pandas as pd
import pytest
import torch

from sdm import CategoricalTensor, StringTensor, TableTensor
from sdm.processing import AlignCategories
from sdm.tensor import EnsembleTable
from sdm.testing import withCUDA


def _table(
    values: list[list[int]],
    *,
    categories: tuple[tuple[str, ...], ...],
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.int32,
) -> TableTensor:
    return TableTensor(
        categorical=CategoricalTensor(
            code=torch.tensor(values, dtype=dtype, device=device),
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


@withCUDA
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_align_categories_keeps_string_vocabularies_column_local(
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    context = _table(
        [[0, 0], [1, 1]],
        categories=(("shared", "left"), ("right", "shared")),
        device=device,
        dtype=dtype,
    )
    query = _table(
        [[0, 0], [1, 1], [2, 2], [-1, -1]],
        categories=(
            ("shared", "right", "left"),
            ("shared", "left", "right"),
        ),
        device=device,
        dtype=dtype,
    )

    output = AlignCategories().fit(context).transform(query)

    assert output.categorical.code.dtype == dtype
    assert torch.equal(
        output.categorical.code,
        torch.tensor(
            [[0, 1], [-1, -1], [1, 0], [-1, -1]],
            dtype=dtype,
            device=device,
        ),
    )


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
def test_align_categories_filters_rare_categories(
    device: torch.device,
) -> None:
    context = _table(
        [[-1], [2], [1], [2], [0], [3], [1], [0]],
        categories=(("alpha", "beta", "gamma", "rare", "unused"),),
        device=device,
    )
    query = _table(
        [[3], [0], [2], [1], [4], [-1]],
        categories=(("gamma", "other", "alpha", "beta", "rare"),),
        device=device,
    )

    processor = AlignCategories(min_frequency=2)
    context_output = processor.fit_transform(context)
    query_output = processor.transform(query)

    assert context_output.categorical.categories[0].tolist() == [
        "alpha",
        "beta",
        "gamma",
    ]
    assert context_output.categorical.code.squeeze(-1).tolist() == [
        -1,
        2,
        1,
        2,
        0,
        -1,
        1,
        0,
    ]
    assert query_output.categorical.code.squeeze(-1).tolist() == [
        1,
        2,
        0,
        -1,
        -1,
        -1,
    ]


@withCUDA
def test_align_categories_filters_per_ensemble_member(
    device: torch.device,
) -> None:
    first = _table(
        [[0], [0], [1]],
        categories=(("red", "blue", "green"),),
        device=device,
    )
    second = _table(
        [[1], [1], [2]],
        categories=(("red", "blue", "green"),),
        device=device,
    )
    context = EnsembleTable.from_tables(
        tables=(first, second),
        member_table_ids=(0, 1),
    )

    output = AlignCategories(min_frequency=2).fit_transform_ensemble(context)

    assert output.table(0).categorical.categories[0].tolist() == ["red"]
    assert output.table(0).categorical.code.squeeze(-1).tolist() == [0, 0, -1]
    assert output.table(1).categorical.categories[0].tolist() == ["blue"]
    assert output.table(1).categorical.code.squeeze(-1).tolist() == [0, 0, -1]


@withCUDA
def test_align_categories_orders_values(
    device: torch.device,
) -> None:
    context = _table(
        [[0], [1], [2], [-1]],
        categories=(("zebra", "éclair", "ant", "unused"),),
        device=device,
    )
    query = _table(
        [[0], [1], [2], [-1]],
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
        categorical=CategoricalTensor(
            code=torch.tensor([[0], [1]], dtype=torch.int32, device=device),
            categories=(torch.tensor([torch.nan, 1.0], device=device),),
        ),
    )
    query = TableTensor(
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
        categorical=CategoricalTensor(
            code=codes,
            categories=(torch.arange(size),),
        ),
    )
    query = TableTensor(
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


# CPU only: PyTorch has no CPU indexing kernels for these dtypes, so the
# unsigned code path is unreachable on CUDA.
@pytest.mark.parametrize("dtype", [torch.uint16, torch.uint32, torch.uint64])
@pytest.mark.parametrize(
    ("codes", "expected_categories", "expected_codes"),
    [
        ([[0], [0], [0]], [10], [0, 0, 0]),
        ([[-1], [-1], [-1]], [], [-1, -1, -1]),
    ],
)
def test_align_categories_drops_unobserved_unsigned_categories(
    dtype: torch.dtype,
    codes: list[list[int]],
    expected_categories: list[int],
    expected_codes: list[int],
) -> None:
    table = TableTensor(
        categorical=CategoricalTensor(
            code=torch.tensor(codes, dtype=torch.int32),
            categories=(torch.tensor([10, 20], dtype=dtype),),
        ),
    )

    output = AlignCategories().fit_transform(table)

    assert output.categorical.categories[0].dtype == dtype
    assert output.categorical.categories[0].tolist() == expected_categories
    assert output.categorical.code.squeeze(-1).tolist() == expected_codes


def test_align_categories_all_missing_context_has_empty_vocabulary() -> None:
    context = _table(
        [[-1], [-1]],
        categories=(("red", "blue"),),
    )
    query = _table(
        [[0], [1], [-1]],
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
    processor = AlignCategories().fit(_table([[0]], categories=(("red",),)))
    query = TableTensor(
        categorical=CategoricalTensor(
            code=torch.tensor([[0]], dtype=torch.int32),
            categories=(torch.tensor([1]),),
        ),
    )

    with pytest.raises(NotImplementedError):
        processor.transform(query)


@withCUDA
def test_align_categories_ensemble_matches_member_fits(
    device: torch.device,
) -> None:
    first_context = _table(
        [[0], [1]],
        categories=(("red", "blue", "green"),),
        device=device,
    )
    second_context = _table(
        [[1], [2]],
        categories=(("red", "blue", "green"),),
        device=device,
    )
    member_table_ids = (1, 0, 1, 0, 0, 1, 1, 0)
    context = EnsembleTable.from_tables(
        tables=(first_context, second_context),
        member_table_ids=member_table_ids,
    )
    query = _table(
        [[0], [1], [2]],
        categories=(("red", "blue", "green"),),
        device=device,
    )
    query_ensemble = EnsembleTable.from_tables(
        tables=(query, query),
        member_table_ids=member_table_ids,
    )
    combined = AlignCategories()
    fitted = AlignCategories().fit_ensemble(context)

    context_output = combined.fit_transform_ensemble(context)
    query_output = combined.transform_ensemble(query_ensemble)
    fitted_query_output = fitted.transform_ensemble(query_ensemble)
    references = [
        AlignCategories().fit(first_context),
        AlignCategories().fit(second_context),
    ]
    context_tables = (first_context, second_context)
    for member_id, table_id in enumerate(member_table_ids):
        reference = references[table_id]
        assert context_output.table(member_id).equal(
            reference.transform(context_tables[table_id])
        )
        expected_query = reference.transform(query)
        assert query_output.table(member_id).equal(expected_query)
        assert fitted_query_output.table(member_id).equal(expected_query)
