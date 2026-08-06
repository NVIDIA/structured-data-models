from typing import Literal, cast

import pandas as pd
import pytest
import torch

from sdm import CategoricalTensor, StringTensor, TableTensor
from sdm.processing import AlignCategories
from sdm.tensor import EnsembleTable
from sdm.tensor.string import _hash_strings, _string_metadata
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


@pytest.mark.parametrize("fullgraph", [False, True])
def test_align_categories_compile(fullgraph: bool) -> None:
    torch._dynamo.reset()
    context = _table(
        [[0], [1], [0]],
        columns=("kind",),
        categories=(("red", "blue", "unused"),),
    )
    query = _table(
        [[0], [1], [2], [-1]],
        columns=("kind",),
        categories=(("green", "blue", "red"),),
    )
    processor = AlignCategories().fit(context)
    transform = torch.compile(
        processor.transform,
        fullgraph=fullgraph,
        backend="eager",
    )

    for _ in range(2):
        output = transform(query)
        assert output.categorical.code.squeeze(-1).tolist() == [-1, 1, 0, -1]
        assert output.categorical.categories[0].tolist() == ["red", "blue"]


def test_align_categories_compile_empty_string_vocabulary() -> None:
    torch._dynamo.reset()
    size = 65
    index = torch.arange(size, dtype=torch.int32)
    columns = (
        "empty",
        "empty_to_value",
        "value_to_empty",
        "mixed",
    )
    context = TableTensor(
        columns={"categorical": columns},
        categorical=CategoricalTensor(
            code=torch.stack((index, index, index, index % 2), dim=1),
            categories=(
                StringTensor.from_list([""] * size),
                StringTensor.from_list([""] * size),
                StringTensor.from_list(["value"] * size),
                StringTensor.from_list(["", "value"]),
            ),
        ),
    )
    query = TableTensor(
        columns={"categorical": columns},
        categorical=CategoricalTensor(
            code=torch.stack((index, index, index, index % 3), dim=1),
            categories=(
                StringTensor.from_list([""] * size),
                StringTensor.from_list(["value"] * size),
                StringTensor.from_list([""] * size),
                StringTensor.from_list(["value", "", "other"]),
            ),
        ),
    )
    processor = AlignCategories().fit(context)
    transform = torch.compile(
        processor.transform,
        fullgraph=True,
        backend="inductor",
    )

    output = transform(query)

    assert output.categorical.code.equal(
        torch.stack(
            (
                torch.zeros_like(index),
                torch.full_like(index, -1),
                torch.full_like(index, -1),
                torch.tensor((1, 0, -1), dtype=torch.int32).repeat(22)[:size],
            ),
            dim=1,
        )
    )


def test_align_categories_compile_chunked_empty_string_vocabulary() -> None:
    torch._dynamo.reset()
    size = 1025
    index = torch.arange(size, dtype=torch.int32)
    empty_categories = StringTensor.from_list([""] * size)
    value_categories = StringTensor.from_list(
        [f"value-{value:04d}" for value in range(size)]
    )
    context = TableTensor(
        columns={"categorical": ("empty", "value")},
        categorical=CategoricalTensor(
            code=torch.stack((index, index), dim=1),
            categories=(empty_categories, value_categories),
        ),
    )
    query = TableTensor(
        columns={"categorical": ("empty", "value")},
        categorical=CategoricalTensor(
            code=torch.stack((index, index), dim=1),
            categories=(
                empty_categories.clone(),
                value_categories.index_select(
                    0,
                    torch.arange(size - 1, -1, -1),
                ),
            ),
        ),
    )
    transform = torch.compile(
        AlignCategories().fit(context).transform,
        fullgraph=True,
        backend="inductor",
    )

    output = transform(query)

    assert output.categorical.code.equal(
        torch.stack((torch.zeros_like(index), index.flip(0)), dim=1)
    )


def test_align_categories_compile_with_aliased_string_backing() -> None:
    backing = StringTensor.from_list(
        ["red", "blue", "green", "x" * (1024 * 1024)]
    )
    category = cast(StringTensor, backing[:3])
    context = TableTensor(
        columns={"categorical": ("kind",)},
        categorical=CategoricalTensor(
            code=torch.arange(3, dtype=torch.int32).unsqueeze(-1),
            categories=(category,),
        ),
    )
    aliased_category = StringTensor(
        data=torch.ops.aten.alias.default(category._data),
        offset=torch.ops.aten.alias.default(category._offset),
        valid=None,
        size=category.size(),
    )
    query = TableTensor(
        columns={"categorical": ("kind",)},
        categorical=CategoricalTensor(
            code=torch.arange(3, dtype=torch.int32).unsqueeze(-1),
            categories=(aliased_category,),
        ),
    )
    processor = AlignCategories().fit(context)
    match_category = cast(StringTensor, processor._match_categories[0][0])
    assert match_category._data.numel() == len("redbluegreen")

    transform = torch.compile(
        lambda table: processor.transform(table),
        fullgraph=True,
        dynamic=True,
        backend="aot_eager",
    )

    output = transform(query)

    assert output.categorical.code.squeeze(-1).tolist() == [0, 1, 2]


def test_align_categories_owns_fitted_string_vocabulary() -> None:
    category = StringTensor.from_list(["red", "blue"])
    context = TableTensor(
        columns={"categorical": ("kind",)},
        categorical=CategoricalTensor(
            code=torch.tensor([[0], [1]], dtype=torch.int32),
            categories=(category,),
        ),
    )
    processor = AlignCategories().fit(context)

    category._data[0] = ord("x")
    query = _table(
        [[0], [1]],
        columns=("kind",),
        categories=(("red", "blue"),),
    )
    output = processor.transform(query)

    assert output.categorical.code.squeeze(-1).tolist() == [0, 1]
    assert output.categorical.categories[0].tolist() == ["red", "blue"]


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


def test_align_categories_ignores_unused_string_vocabulary() -> None:
    size = 2_048
    categories = tuple(f"value-{i}" for i in range(size))
    context = _table(
        [[i] for i in range(size)],
        columns=("value",),
        categories=(categories,),
    )
    query = _table(
        [[size // 2]],
        columns=("value",),
        categories=(tuple(reversed(categories)),),
    )

    output = AlignCategories().fit(context).transform(query)

    assert output.categorical.code.item() == size // 2 - 1


@withCUDA
def test_align_categories_reuses_duplicate_string_codes(
    device: torch.device,
) -> None:
    size = 256
    categories = tuple(f"value-{i:03d}-" + "x" * 48 for i in range(size))
    context = _table(
        [[i] for i in range(size)],
        columns=("value",),
        categories=(categories,),
        device=device,
    )
    query = _table(
        [[0]] * 1_024,
        columns=("value",),
        categories=(tuple(reversed(categories)),),
        device=device,
    )
    processor = AlignCategories().fit(context)
    transform = torch.compile(
        lambda table: processor.transform(table),
        fullgraph=True,
        # Category vocabularies are schema state rather than batch dimensions.
        dynamic=False,
        backend="aot_eager",
    )

    output = transform(query)

    assert output.categorical.code.squeeze(-1).equal(
        torch.full(
            (1_024,),
            size - 1,
            dtype=torch.int32,
            device=device,
        )
    )

    different_shape = _table(
        [[0], [1], [1]],
        columns=("value",),
        categories=(("missing-" + "y" * 67, categories[3]),),
        device=device,
    )
    output = transform(different_shape)

    assert output.categorical.code.squeeze(-1).tolist() == [-1, 3, 3]


@withCUDA
def test_align_categories_handles_duplicate_nonempty_vocabularies(
    device: torch.device,
) -> None:
    size = 2_000
    category = StringTensor.from_list(
        ["duplicate-" + "x" * 54] * size,
        device=device,
    )
    code = torch.arange(size, dtype=torch.int32, device=device).unsqueeze(-1)
    table = TableTensor(
        columns={"categorical": ("value",)},
        categorical=CategoricalTensor(code=code, categories=(category,)),
    )
    processor = AlignCategories().fit(table)
    transform = torch.compile(
        lambda table: processor.transform(table),
        fullgraph=True,
        dynamic=True,
        backend="aot_eager",
    )

    output = transform(table)

    assert output.categorical.code.squeeze(-1).equal(
        torch.zeros_like(code[:, 0])
    )


def test_align_categories_handles_empty_string_vocabularies() -> None:
    size = 2_000
    category = StringTensor.from_list([""] * size)
    code = torch.arange(size, dtype=torch.int32).unsqueeze(-1)
    table = TableTensor(
        columns={"categorical": ("value",)},
        categorical=CategoricalTensor(code=code, categories=(category,)),
    )

    output = AlignCategories().fit(table).transform(table)

    assert output.categorical.code.squeeze(-1).equal(
        torch.zeros_like(code[:, 0])
    )

    context = _table(
        [[0]],
        columns=("value",),
        categories=(("nonempty",),),
    )
    query = _table(
        [[0]],
        columns=("value",),
        categories=(("",),),
    )
    assert (
        AlignCategories().fit(context).transform(query).categorical.code.item()
        == -1
    )


def test_align_categories_handles_strided_string_vocabulary() -> None:
    context = _table(
        [[0], [1], [2]],
        columns=("value",),
        categories=(("a", "", "z"),),
    )
    category = cast(
        StringTensor,
        StringTensor.from_list(
            ["junk", "a", "unused", "", "other", "a", "ignored"]
        )[1::2],
    )
    query = TableTensor(
        columns={"categorical": ("value",)},
        categorical=CategoricalTensor(
            code=torch.tensor([[0], [1], [2], [-1]], dtype=torch.int32),
            categories=(category,),
        ),
    )

    output = AlignCategories().fit(context).transform(query)

    assert output.categorical.code.squeeze(-1).tolist() == [0, 1, 0, -1]


def test_align_categories_ignores_retained_string_backing() -> None:
    context = _table(
        [[0]],
        columns=("value",),
        categories=(("match",),),
    )
    category = cast(
        StringTensor,
        StringTensor.from_list(["x" * (1 << 20), "match"])[1:],
    )
    query = TableTensor(
        columns={"categorical": ("value",)},
        categorical=CategoricalTensor(
            code=torch.zeros((1, 1), dtype=torch.int32),
            categories=(category,),
        ),
    )
    processor = AlignCategories().fit(context)
    transform = torch.compile(
        lambda table: processor.transform(table),
        fullgraph=True,
        backend="aot_eager",
    )

    output = transform(query)

    assert output.categorical.code.item() == 0


def test_align_categories_handles_unmatched_long_strings() -> None:
    context = _table(
        [[0]],
        columns=("value",),
        categories=(("short",),),
    )
    query = _table(
        [[0]],
        columns=("value",),
        categories=(("x" * (1 << 16),),),
    )
    processor = AlignCategories().fit(context)
    transform = torch.compile(
        lambda table: processor.transform(table),
        fullgraph=True,
        backend="aot_eager",
    )

    output = transform(query)

    assert output.categorical.code.item() == -1


@withCUDA
def test_align_categories_verifies_string_hash_collisions(
    device: torch.device,
) -> None:
    value1 = "\x01\x00\x1c\x00F\x00\x1c\x00\x01"
    value2 = "\x00\x08\x008\x008\x00\x08\x00"
    collision = StringTensor.from_list((value1, value2), device=device)
    hashes = _hash_strings(collision._data, _string_metadata(collision))
    assert value1 != value2
    assert hashes[0].equal(hashes[1])
    context = _table(
        [[0], [1], [2]],
        columns=("value",),
        categories=((value1, value2, value2),),
        device=device,
    )
    query = _table(
        [[0], [1]],
        columns=("value",),
        categories=((value2, value1),),
        device=device,
    )
    processor = AlignCategories().fit(context)
    transform = torch.compile(
        lambda table: processor.transform(table),
        fullgraph=True,
        dynamic=True,
        backend="aot_eager",
    )

    output = transform(query)

    assert output.categorical.code.squeeze(-1).tolist() == [1, 0]


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


@pytest.mark.parametrize("fit_strings", [False, True])
def test_align_categories_rejects_changed_category_value_type(
    fit_strings: bool,
) -> None:
    string_table = _table(
        [[0]],
        columns=("kind",),
        categories=(("red",),),
    )
    numeric_table = TableTensor(
        columns={"categorical": ("kind",)},
        categorical=CategoricalTensor(
            code=torch.tensor([[0]], dtype=torch.int32),
            categories=(torch.tensor([1]),),
        ),
    )
    context, query = (
        (string_table, numeric_table)
        if fit_strings
        else (numeric_table, string_table)
    )
    processor = AlignCategories().fit(context)

    with pytest.raises(NotImplementedError, match="category value types"):
        processor.transform(query)


@withCUDA
def test_align_categories_ensemble_matches_member_fits(
    device: torch.device,
) -> None:
    first_context = _table(
        [[0], [1]],
        columns=("kind",),
        categories=(("red", "blue", "green"),),
        device=device,
    )
    second_context = _table(
        [[1], [2]],
        columns=("kind",),
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
        columns=("kind",),
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


def test_align_categories_ensemble_state_follows_logical_members() -> None:
    first_context = _table(
        [[0]],
        columns=("kind",),
        categories=(("red", "blue"),),
    )
    second_context = _table(
        [[1]],
        columns=("kind",),
        categories=(("red", "blue"),),
    )
    context = EnsembleTable.from_tables(
        tables=(first_context, second_context),
        member_table_ids=(0, 1),
    )
    query = _table(
        [[0], [1]],
        columns=("kind",),
        categories=(("red", "blue"),),
    )
    processor = AlignCategories().fit_ensemble(context)

    output = processor.transform_ensemble(EnsembleTable(query, num_members=2))

    assert output.table(0).categorical.code.squeeze(-1).tolist() == [0, -1]
    assert output.table(0).categorical.categories[0].tolist() == ["red"]
    assert output.table(1).categorical.code.squeeze(-1).tolist() == [-1, 0]
    assert output.table(1).categorical.categories[0].tolist() == ["blue"]
