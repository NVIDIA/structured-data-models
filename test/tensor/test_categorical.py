import io
import warnings
from typing import cast

import pyarrow as pa
import pytest
import torch

from sdm import CategoricalTensor, StringTensor
from sdm.tensor.categorical import _make_categorical_tensor
from sdm.testing import onlyCUDA, withCUDA


def test_to_copy_string_categories() -> None:
    data = torch.randint(0, 2, size=(10, 2))
    categories = (
        StringTensor.from_list(["USA", "GERMANY"]),
        StringTensor.from_list(["enterprise", "startup"]),
    )
    tensor = CategoricalTensor(data, categories)

    out = tensor.clone()
    assert isinstance(out, CategoricalTensor)
    for out_category, category in zip(out.categories, categories):
        assert out_category.tolist() == category.tolist()

    out = tensor.to(torch.int32)
    assert isinstance(out, CategoricalTensor)
    assert out.dtype == torch.int32
    for out_category, category in zip(out.categories, categories):
        assert out_category.tolist() == category.tolist()


def test_tensor_flatten_round_trip() -> None:
    tensor = CategoricalTensor(
        code=torch.tensor([[0, 1], [1, 0]], dtype=torch.int32),
        categories=(
            StringTensor.from_list(["a", "b"]),
            torch.tensor([10, 20]),
        ),
    )
    names, context = tensor.__tensor_flatten__()

    out = CategoricalTensor.__tensor_unflatten__(
        {name: getattr(tensor, name) for name in names},
        context,
        tensor.size(),
        tensor.stride(),
    )

    assert type(out) is type(tensor)
    assert out.size() == tensor.size()
    assert out.stride() == tensor.stride()
    assert out.code.equal(tensor.code)
    assert out.categories[0].tolist() == tensor.categories[0].tolist()
    assert out.categories[1].equal(tensor.categories[1])

    tensor = cast(CategoricalTensor, tensor[..., 1:])
    names, context = tensor.__tensor_flatten__()
    out = CategoricalTensor.__tensor_unflatten__(
        {name: getattr(tensor, name) for name in names},
        context,
        tensor.size(),
        tensor.stride(),
    )

    assert out.code.equal(tensor.code)
    assert len(out.categories) == 1
    assert out.categories[0].equal(tensor.categories[0])


def test_category_uses_logical_column_index() -> None:
    categories = tuple(torch.tensor([10 * index]) for index in range(4))
    tensor = CategoricalTensor(
        code=torch.zeros((2, 4), dtype=torch.int32),
        categories=categories,
    )
    view = cast(CategoricalTensor, tensor[..., 1:3])

    assert view.category(0) is categories[1]
    assert view.category(-1) is categories[2]
    with pytest.raises(IndexError):
        view.category(2)

    compiled = torch.compile(
        lambda value: value.category(-1),
        fullgraph=True,
        backend="eager",
    )
    assert compiled(view).equal(categories[2])

    compiled_view = torch.compile(
        lambda value: value[..., 1:3].category(-1),
        fullgraph=True,
        backend="eager",
    )
    assert compiled_view(tensor).equal(categories[2])


def test_save_load_preserves_sliced_category_topology() -> None:
    tensor = cast(
        CategoricalTensor,
        CategoricalTensor(
            code=torch.zeros((2, 4), dtype=torch.int32),
            categories=tuple(torch.tensor([10 * i]) for i in range(4)),
        )[..., 1:3],
    )
    before = torch.as_strided(tensor, (2, 2), (2, 1), 1)
    assert type(before) is torch.Tensor

    buffer = io.BytesIO()
    torch.save(tensor, buffer)
    buffer.seek(0)
    out = torch.load(buffer, weights_only=False)

    assert isinstance(out, CategoricalTensor)
    assert out.categories == tensor.categories
    after = torch.as_strided(out, (2, 2), (2, 1), 1)
    assert type(after) is torch.Tensor
    assert after.equal(before)


def test_materialized_slice_resets_category_topology() -> None:
    tensor = cast(
        CategoricalTensor,
        CategoricalTensor(
            code=torch.zeros((2, 4), dtype=torch.int32),
            categories=tuple(torch.tensor([10 * i]) for i in range(4)),
        )[..., 1:],
    )
    materialized = (tensor.clone(), tensor.contiguous())
    identity_as_strided = torch.compile(
        lambda value: torch.as_strided(
            value,
            (2, 3),
            (3, 1),
            0,
        ),
        fullgraph=True,
        backend="eager",
    )

    for out in materialized:
        assert isinstance(out, CategoricalTensor)
        assert out.categories == tensor.categories
        identity = identity_as_strided(out)
        assert isinstance(identity, CategoricalTensor)
        assert identity.categories == tensor.categories


def test_compile() -> None:
    tensor = CategoricalTensor(
        code=torch.tensor([[0], [1]], dtype=torch.int32),
        categories=(StringTensor.from_list(["a", "b"]),),
    )
    compiled = torch.compile(
        lambda value: value.view(-1, 1),
        fullgraph=True,
        backend="eager",
    )

    for _ in range(2):
        out = compiled(tensor)
        assert isinstance(out, CategoricalTensor)
        assert out.tolist() == tensor.tolist()
        assert torch._C._is_alias_of(  # ty: ignore[unresolved-attribute]
            out,
            tensor,
        )

    compiled_view = torch.compile(
        lambda value: value.unsqueeze(0),
        fullgraph=True,
        backend="aot_eager",
    )
    for _ in range(2):
        out = compiled_view(tensor)
        assert isinstance(out, CategoricalTensor)
        assert out.size() == (1, 2, 1)
        assert out.tolist() == [tensor.tolist()]
        assert torch._C._is_alias_of(  # ty: ignore[unresolved-attribute]
            out,
            tensor,
        )

    code = torch.tensor([[0], [1]], dtype=torch.int32)
    category = StringTensor.from_list(["a", "b"])
    construct = torch.compile(
        lambda value, categories: _make_categorical_tensor(
            value + 0,
            (categories,),
        ),
        fullgraph=True,
        backend="eager",
    )

    for _ in range(2):
        out = construct(code, category)
        assert isinstance(out, CategoricalTensor)
        assert out.code.equal(code)
        assert out.categories[0].tolist() == category.tolist()

    clone = torch.compile(
        lambda value: value.clone(),
        fullgraph=True,
        backend="aot_eager",
    )
    out = clone(tensor)
    assert isinstance(out, CategoricalTensor)
    assert out.tolist() == tensor.tolist()

    convert = torch.compile(
        lambda value: value.to(torch.int64),
        fullgraph=True,
        backend="aot_eager",
    )
    out = convert(tensor)
    assert isinstance(out, CategoricalTensor)
    assert out.dtype == torch.int64
    assert out.tolist() == tensor.tolist()

    tensor = CategoricalTensor(
        torch.tensor([[0, 1, 2, 3]], dtype=torch.int32),
        tuple(torch.arange(4) + 10 * i for i in range(4)),
    )
    slice_columns = torch.compile(
        lambda value: value[..., 1:3],
        fullgraph=True,
        backend="aot_eager",
    )
    out = slice_columns(tensor)
    assert isinstance(out, CategoricalTensor)
    assert out.categories == tensor.categories[1:3]
    assert torch._C._is_alias_of(  # ty: ignore[unresolved-attribute]
        out,
        tensor,
    )

    tensor = CategoricalTensor(
        torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]], dtype=torch.int32),
        tensor.categories,
    )
    chained_view = torch.compile(
        lambda value: torch.as_strided(
            value[..., ::2],
            (2, 1),
            (4, 2),
            2,
        ),
        fullgraph=True,
        backend="aot_eager",
    )
    out = chained_view(tensor)
    assert isinstance(out, CategoricalTensor)
    assert out.categories == (tensor.categories[2],)
    assert torch._C._is_alias_of(  # ty: ignore[unresolved-attribute]
        out,
        tensor,
    )

    slice_alternating_columns = torch.compile(
        lambda value: value[..., ::2],
        fullgraph=True,
        backend="aot_eager",
    )
    out = slice_alternating_columns(tensor)
    assert isinstance(out, CategoricalTensor)
    assert out.categories == tensor.categories[::2]
    assert torch._C._is_alias_of(  # ty: ignore[unresolved-attribute]
        out,
        tensor,
    )

    split_columns = torch.compile(
        lambda value: value.split(2, dim=-1),
        fullgraph=True,
        backend="aot_eager",
    )
    out_list = split_columns(tensor)
    assert all(isinstance(value, CategoricalTensor) for value in out_list)
    assert [value.categories for value in out_list] == [
        tensor.categories[:2],
        tensor.categories[2:],
    ]
    assert all(
        torch._C._is_alias_of(tensor, value)  # ty: ignore[unresolved-attribute]
        for value in out_list
    )

    repeat_column = torch.compile(
        lambda value: torch.as_strided(value, (1, 4), (4, 0), 1),
        fullgraph=True,
        backend="aot_eager",
    )
    out = repeat_column(tensor)
    assert isinstance(out, CategoricalTensor)
    assert out.categories == 4 * (tensor.categories[1],)
    assert torch._C._is_alias_of(  # ty: ignore[unresolved-attribute]
        out,
        tensor,
    )


def test_reject_nullable() -> None:
    with pytest.raises(ValueError, match="null values"):
        CategoricalTensor(
            code=torch.tensor([[0], [-1]]),
            categories=(StringTensor.from_list(["a", None]),),
        )


def test_to_copy() -> None:
    data = torch.tensor([[0, -1, 2], [2, 1, 0]])
    categories = tuple(torch.arange(3) for _ in range(data.size(-1)))
    tensor = CategoricalTensor(data, categories)

    out = tensor.to(torch.int32)
    assert isinstance(out, CategoricalTensor)
    assert out.dtype == torch.int32
    assert out.code.dtype == torch.int32
    for out_category, category in zip(out.categories, tensor.categories):
        assert out_category.equal(category)
        assert out_category.data_ptr() != category.data_ptr()

    out = tensor.to(torch.float32)
    assert not isinstance(out, CategoricalTensor)
    assert out.dtype == torch.float32


def test_from_arrow_string_values() -> None:
    tensor = CategoricalTensor.from_arrow(
        pa.array(["b", "a", None, "b"]),
    )

    assert tensor.dtype == torch.int32
    assert tensor.code.equal(torch.tensor([[0], [1], [-1], [0]]))
    assert tensor.categories[0].tolist() == ["b", "a"]
    assert tensor.to_arrow().column(0).chunk(0).dictionary.type == pa.string()


def test_from_arrow_chunked_values() -> None:
    tensor = CategoricalTensor.from_arrow(
        pa.chunked_array([pa.array(["b", None]), pa.array(["a", "b"])]),
    )

    assert tensor.dtype == torch.int32
    assert tensor.code.equal(torch.tensor([[0], [-1], [1], [0]]))
    assert tensor.categories[0].tolist() == ["b", "a"]
    assert (
        tensor.to_arrow().column(0).chunk(0).dictionary.type
        == pa.large_string()
    )


def test_from_arrow_chunked_large_string_values() -> None:
    tensor = CategoricalTensor.from_arrow(
        pa.chunked_array(
            [
                pa.array(["b", None], type=pa.large_string()),
                pa.array(["a", "b"], type=pa.large_string()),
            ]
        ),
    )

    assert tensor.code.equal(torch.tensor([[0], [-1], [1], [0]]))
    assert tensor.categories[0].tolist() == ["b", "a"]
    assert tensor.to_arrow().to_pydict() == {
        "0": ["b", None, "a", "b"],
    }
    assert (
        tensor.to_arrow().column(0).chunk(0).dictionary.type
        == pa.large_string()
    )


def test_from_arrow_chunked_dictionary_string_values() -> None:
    tensor = CategoricalTensor.from_arrow(
        pa.chunked_array(
            [
                pa.DictionaryArray.from_arrays([0, 1], ["b", "a"]),
                pa.DictionaryArray.from_arrays([0, 1], ["a", "c"]),
            ]
        ),
    )

    assert tensor.to_arrow().to_pydict() == {
        "0": ["b", "a", "a", "c"],
    }
    assert (
        tensor.to_arrow().column(0).chunk(0).dictionary.type
        == pa.large_string()
    )


def test_from_arrow_numeric_values() -> None:
    tensor = CategoricalTensor.from_arrow(
        pa.array([10, 20, None, 10], type=pa.int32()),
    )

    assert tensor.dtype == torch.int32
    assert tensor.code.equal(torch.tensor([[0], [1], [-1], [0]]))
    assert tensor.categories[0].equal(torch.tensor([10, 20]))


def test_from_arrow_all_missing_values() -> None:
    tensor = CategoricalTensor.from_arrow(
        pa.array([None, None], type=pa.string()),
    )

    assert tensor.dtype == torch.int32
    assert tensor.code.equal(torch.tensor([[-1], [-1]]))
    assert tensor.categories[0].numel() == 0


def test_from_arrow_dtype() -> None:
    tensor = CategoricalTensor.from_arrow(
        pa.array(["b", "a", None]),
        dtype=torch.int64,
    )

    assert tensor.dtype == torch.int64
    assert tensor.code.equal(torch.tensor([[0], [1], [-1]]))


def test_from_arrow_small_dictionary_indices() -> None:
    array = pa.DictionaryArray.from_arrays(
        pa.array([0, 1], type=pa.int8()),
        pa.array(["b", "a"]),
    )

    tensor = CategoricalTensor.from_arrow(array)

    assert tensor.dtype == torch.int32
    assert tensor.code.equal(torch.tensor([[0], [1]]))


@pytest.mark.parametrize("dtype", [torch.int8, torch.float32])
def test_from_arrow_rejects_invalid_dtype(dtype: torch.dtype) -> None:
    with pytest.raises(
        ValueError,
        match=r"Expected .code. in .CategoricalTensor.",
    ):
        CategoricalTensor.from_arrow(pa.array(["b", "a"]), dtype=dtype)


@withCUDA
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_from_arrow_does_not_warn_on_readonly_numpy(
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        CategoricalTensor.from_arrow(
            pa.array([10, 20, None, 10], type=pa.int32()),
            dtype=dtype,
            device=device,
        )

    assert not any(
        "NumPy array is not writable" in str(warning.message)
        for warning in rec
    )


@onlyCUDA
def test_from_arrow_cuda() -> None:
    tensor = CategoricalTensor.from_arrow(
        pa.array(["b", "a", None, "b"]),
        device="cuda",
    )

    assert tensor.device.type == "cuda"
    assert tensor.code.equal(
        torch.tensor(
            [[0], [1], [-1], [0]],
            dtype=torch.int32,
            device=tensor.device,
        )
    )
    assert tensor.categories[0].device.type == "cuda"
    assert tensor.categories[0].tolist() == ["b", "a"]

    tensor = CategoricalTensor.from_arrow(
        pa.array([10, 20, None, 10], type=pa.int32()),
        device="cuda",
    )

    assert tensor.device.type == "cuda"
    assert tensor.code.equal(
        torch.tensor(
            [[0], [1], [-1], [0]],
            dtype=torch.int32,
            device=tensor.device,
        )
    )
    assert tensor.categories[0].device.type == "cuda"
    assert tensor.categories[0].equal(
        torch.tensor([10, 20], dtype=torch.int32, device=tensor.device)
    )


def test_tolist() -> None:
    tensor = CategoricalTensor(
        code=torch.tensor(
            [
                [[0, 1], [-1, 0]],
                [[1, -1], [0, 1]],
            ],
            dtype=torch.int32,
        ),
        categories=(
            torch.tensor([10, 20]),
            torch.tensor([30, 40]),
        ),
    )

    assert tensor.tolist() == [
        [[10, 40], [None, 30]],
        [[20, None], [10, 40]],
    ]


def test_to_arrow() -> None:
    tensor = CategoricalTensor(
        code=torch.tensor(
            [
                [[0, 1], [-1, 0]],
                [[1, -1], [0, 1]],
            ],
            dtype=torch.int32,
        ),
        categories=(
            StringTensor.from_list(["US", "CA"]),
            torch.tensor([10, 20]),
        ),
    )

    table = tensor.to_arrow()
    assert table.column_names == ["0", "1"]
    assert pa.types.is_dictionary(table["0"].type)
    assert pa.types.is_dictionary(table["1"].type)
    assert table.to_pydict() == {
        "0": ["US", None, "CA", "US"],
        "1": [20, 10, None, 20],
    }


@onlyCUDA
def test_from_cudf_string_values() -> None:
    cudf = pytest.importorskip("cudf")

    tensor = CategoricalTensor.from_cudf(
        cudf.Series(["b", "a", None, "b"]),
    )

    assert tensor.is_cuda
    assert tensor.code.equal(
        torch.tensor([[0], [1], [-1], [0]], device=tensor.device)
    )
    assert tensor.categories[0].is_cuda
    assert tensor.categories[0].tolist() == ["b", "a"]


@onlyCUDA
def test_from_cudf_numeric_values() -> None:
    cudf = pytest.importorskip("cudf")

    tensor = CategoricalTensor.from_cudf(
        cudf.Series([10, 20, None, 10], dtype="int32"),
    )

    assert tensor.is_cuda
    assert tensor.code.equal(
        torch.tensor([[0], [1], [-1], [0]], device=tensor.device)
    )
    assert tensor.categories[0].is_cuda
    assert tensor.categories[0].tolist() == [10, 20]


@onlyCUDA
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize(
    ("values", "categories", "expected_code"),
    [
        (
            ["b", None, "a", "b"],
            ["unused", "a", "b"],
            [[2], [-1], [1], [2]],
        ),
        (
            [20, None, 10, 20],
            [30, 10, 20],
            [[2], [-1], [1], [2]],
        ),
        (
            [None, None],
            ["a", "b"],
            [[-1], [-1]],
        ),
    ],
)
def test_from_cudf_categorical_values(
    values: list[str | int | None],
    categories: list[str | int],
    expected_code: list[list[int]],
    dtype: torch.dtype,
) -> None:
    cudf = pytest.importorskip("cudf")
    series = cudf.Series(
        values,
        dtype=cudf.CategoricalDtype(categories=categories),
    )

    tensor = CategoricalTensor.from_cudf(series, dtype=dtype)

    assert tensor.is_cuda
    assert tensor.dtype == dtype
    assert tensor.code.equal(torch.tensor(expected_code, device=tensor.device))
    assert tensor.categories[0].is_cuda
    assert tensor.categories[0].tolist() == categories


@onlyCUDA
def test_from_cudf_all_missing_values() -> None:
    cudf = pytest.importorskip("cudf")

    tensor = CategoricalTensor.from_cudf(
        cudf.Series([None, None], dtype="int32"),
    )

    assert tensor.is_cuda
    assert tensor.code.equal(torch.tensor([[-1], [-1]], device=tensor.device))
    assert tensor.categories[0].is_cuda
    assert tensor.categories[0].numel() == 0


@onlyCUDA
def test_to_cudf() -> None:
    cudf = pytest.importorskip("cudf")

    tensor = CategoricalTensor(
        code=torch.tensor(
            [
                [[0, 1], [-1, 0]],
                [[1, -1], [0, 1]],
            ],
            dtype=torch.int32,
            device="cuda",
        ),
        categories=(
            StringTensor.from_list(["US", "CA"], device="cuda"),
            torch.tensor([10, 20], device="cuda"),
        ),
    )

    df = tensor.to_cudf()
    assert df.columns.tolist() == ["0", "1"]
    assert isinstance(df["0"].dtype, cudf.CategoricalDtype)
    assert isinstance(df["1"].dtype, cudf.CategoricalDtype)
    assert df.to_arrow().to_pydict() == {
        "0": ["US", None, "CA", "US"],
        "1": [20, 10, None, 20],
    }


def test_view_ops() -> None:
    data = torch.randint(0, 4, (2, 3, 4))
    categories = tuple(torch.arange(4) for _ in range(data.size(-1)))
    tensor = CategoricalTensor(data, categories)

    out = tensor.view(6, 4)
    assert isinstance(out, CategoricalTensor)
    assert out.size() == (6, 4)
    assert torch._C._is_alias_of(  # ty: ignore[unresolved-attribute]
        out,
        tensor,
    )

    out = tensor.view(-1)
    assert not isinstance(out, CategoricalTensor)
    assert out.size() == (24,)

    out = tensor.unsqueeze(1)
    assert isinstance(out, CategoricalTensor)
    assert out.size() == (2, 1, 3, 4)

    out = tensor.unsqueeze(1).squeeze(1)
    assert isinstance(out, CategoricalTensor)
    assert out.size() == (2, 3, 4)

    out = tensor.unsqueeze(-1)
    assert not isinstance(out, CategoricalTensor)
    assert out.size() == (2, 3, 4, 1)

    out = tensor.transpose(0, 1)
    assert isinstance(out, CategoricalTensor)
    assert out.size() == (3, 2, 4)

    out = tensor.permute(2, 0, 1)
    assert not isinstance(out, CategoricalTensor)
    assert out.size() == (4, 2, 3)


def test_slicing_ops() -> None:
    data = torch.randint(0, 4, (2, 3, 4))
    categories = tuple(torch.arange(4) for _ in range(data.size(-1)))
    tensor = CategoricalTensor(data, categories)

    out = tensor[:, 1:]
    assert isinstance(out, CategoricalTensor)
    assert out.size() == (2, 2, 4)
    assert out.categories == tensor.categories

    out = tensor[..., 1::2]
    assert isinstance(out, CategoricalTensor)
    assert out.size() == (2, 3, 2)
    assert out.categories == categories[1::2]

    out = tensor.narrow(-1, 1, 2)
    assert isinstance(out, CategoricalTensor)
    assert out.size() == (2, 3, 2)
    assert out.categories == categories[1:3]

    out = tensor[0]
    assert isinstance(out, CategoricalTensor)
    assert out.size() == (3, 4)
    assert out.categories == tensor.categories

    out = tensor[..., 0]
    assert not isinstance(out, CategoricalTensor)
    assert out.size() == (2, 3)


def test_as_strided_category_positions() -> None:
    tensor = CategoricalTensor(
        torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]], dtype=torch.int32),
        tuple(torch.arange(4) + 10 * i for i in range(4)),
    )

    out = torch.as_strided(tensor, (1, 4), (4, 0), 1)
    assert isinstance(out, CategoricalTensor)
    assert out.categories == 4 * (tensor.categories[1],)

    out = torch.as_strided(tensor, (2, 2), (1, 1), 0)
    assert not isinstance(out, CategoricalTensor)
    assert out.equal(torch.as_strided(tensor.code, (2, 2), (1, 1), 0))

    out = torch.as_strided(tensor, (1, 6), (4, 0), 1)
    assert not isinstance(out, CategoricalTensor)
    assert out.equal(torch.as_strided(tensor.code, (1, 6), (4, 0), 1))

    stepped = tensor[..., ::2]
    out = torch.as_strided(stepped, (2, 1), (4, 2), 2)
    assert isinstance(out, CategoricalTensor)
    assert out.categories == (tensor.categories[2],)

    code = torch.as_strided(
        torch.tensor([0, 1, 2, 3], dtype=torch.int32),
        (2, 2),
        (1, 1),
    )
    irregular = CategoricalTensor(code, tensor.categories[:2])
    out = torch.as_strided(irregular, (2, 2), (2, 1), 0)
    assert not isinstance(out, CategoricalTensor)


def test_split_ops() -> None:
    data = torch.randint(0, 4, (2, 3, 4))
    categories = tuple(torch.arange(4) for _ in range(data.size(-1)))
    tensor = CategoricalTensor(data, categories)

    out = cast(tuple[CategoricalTensor], tensor.unbind(0))
    assert all(isinstance(item, CategoricalTensor) for item in out)
    assert [item.size() for item in out] == 2 * [(3, 4)]
    assert all(item.categories == tensor.categories for item in out)
    assert all(
        torch._C._is_alias_of(tensor, item)  # ty: ignore[unresolved-attribute]
        for item in out
    )

    out = tensor.unbind(-1)
    assert all(not isinstance(item, CategoricalTensor) for item in out)
    assert [item.size() for item in out] == 4 * [(2, 3)]

    out = tensor.split(2, dim=-1)
    assert all(isinstance(item, CategoricalTensor) for item in out)
    assert [item.categories for item in out] == [
        categories[:2],
        categories[2:],
    ]
    assert all(
        torch._C._is_alias_of(tensor, item)  # ty: ignore[unresolved-attribute]
        for item in out
    )

    out = tensor.split([1, 3], dim=-1)
    assert all(isinstance(item, CategoricalTensor) for item in out)
    assert [item.categories for item in out] == [
        categories[:1],
        categories[1:],
    ]
    assert all(
        torch._C._is_alias_of(tensor, item)  # ty: ignore[unresolved-attribute]
        for item in out
    )


def test_index_ops() -> None:
    data = torch.randint(0, 4, (2, 3, 4))
    categories = tuple(torch.arange(4) for _ in range(data.size(-1)))
    tensor = CategoricalTensor(data, categories)

    out = tensor.index_select(1, torch.tensor([2, 0]))
    assert isinstance(out, CategoricalTensor)
    assert out.size() == (2, 2, 4)
    assert out.categories == tensor.categories

    out = tensor.index_select(-1, torch.tensor([2, 0]))
    assert isinstance(out, CategoricalTensor)
    assert out.size() == (2, 3, 2)
    assert out.categories == (categories[2], categories[0])

    out = tensor[[1, 0]]
    assert isinstance(out, CategoricalTensor)
    assert out.size() == (2, 3, 4)
    assert out.categories == tensor.categories

    out = tensor[..., [2, 0]]
    assert isinstance(out, CategoricalTensor)
    assert out.size() == (2, 3, 2)
    assert out.categories == (categories[2], categories[0])

    out = tensor[..., [True, False, True, False]]
    assert isinstance(out, CategoricalTensor)
    assert out.size() == (2, 3, 2)
    assert out.categories == (categories[0], categories[2])

    out = tensor[[1, 0], :, [2, 1]]
    assert not isinstance(out, CategoricalTensor)
    assert out.size() == (2, 3)

    mask = torch.zeros_like(data, dtype=torch.bool)
    mask[0, :2] = True
    out = tensor[mask]
    assert not isinstance(out, CategoricalTensor)
    assert out.size() == (8,)


def test_cat_stack() -> None:
    data1 = torch.randint(0, 4, (2, 3, 4))
    data2 = torch.randint(0, 4, (2, 3, 4))
    categories = tuple(torch.arange(4) for _ in range(data1.size(-1)))
    tensor1 = CategoricalTensor(data1, categories)
    tensor2 = CategoricalTensor(data2, categories)

    out = torch.cat([tensor1, tensor2], dim=0)
    assert isinstance(out, CategoricalTensor)
    assert out.size() == (4, 3, 4)
    assert out.categories == categories

    out = torch.cat([tensor1, data2], dim=0)
    assert not isinstance(out, CategoricalTensor)
    assert out.size() == (4, 3, 4)

    out = torch.cat([tensor1, tensor2], dim=-1)
    assert isinstance(out, CategoricalTensor)
    assert out.size() == (2, 3, 8)
    assert out.categories == categories + categories

    out = torch.stack([tensor1, tensor2], dim=1)
    assert isinstance(out, CategoricalTensor)
    assert out.size() == (2, 2, 3, 4)
    assert out.categories == categories

    out = torch.stack([tensor1, data2], dim=1)
    assert not isinstance(out, CategoricalTensor)
    assert out.size() == (2, 2, 3, 4)

    out = torch.stack([tensor1, tensor2], dim=-1)
    assert not isinstance(out, CategoricalTensor)
    assert out.size() == (2, 3, 4, 2)


def test_pin_memory() -> None:
    data = torch.tensor([[0, -1, 2], [2, 1, 0]])
    categories = tuple(torch.arange(3) for _ in range(data.size(-1)))
    tensor = CategoricalTensor(data, categories)

    assert not tensor.is_pinned()


@onlyCUDA
def test_pin_memory_cuda() -> None:
    data = torch.tensor([[0, -1, 2], [2, 1, 0]])
    categories = tuple(torch.arange(3) for _ in range(data.size(-1)))
    tensor = CategoricalTensor(data, categories)

    assert tensor.pin_memory().is_pinned()


def test_equal_allclose() -> None:
    tensor1 = CategoricalTensor(
        code=torch.tensor([[0], [1]], dtype=torch.int32),
        categories=(StringTensor.from_list(["a", "b"]),),
    )

    tensor2 = CategoricalTensor(
        code=torch.tensor([[0], [1]], dtype=torch.int32),
        categories=(StringTensor.from_list(["a", "b"]),),
    )
    assert tensor1.equal(tensor2)
    assert tensor1.allclose(tensor2)

    tensor3 = CategoricalTensor(
        code=torch.tensor([[0], [1]], dtype=torch.int32),
        categories=(StringTensor.from_list(["x", "y"]),),
    )
    assert not tensor1.equal(tensor3)
    assert not tensor1.allclose(tensor3)

    tensor4 = CategoricalTensor(
        code=torch.tensor([[1], [1]], dtype=torch.int32),
        categories=(StringTensor.from_list(["a", "b"]),),
    )
    assert not tensor1.allclose(tensor4)
    assert not tensor1.allclose(tensor4)


def test_share_memory() -> None:
    data = torch.tensor([[0, -1, 2], [2, 1, 0]])
    categories = tuple(torch.arange(3) for _ in range(data.size(-1)))
    tensor = CategoricalTensor(data, categories)

    assert not tensor.is_shared()
    try:
        tensor.share_memory_()
        assert tensor.is_shared()
    except RuntimeError:
        pass


def test_isnan() -> None:
    data = torch.tensor([[0, -1, 2], [-2, 1, 0]])
    categories = tuple(torch.arange(3) for _ in range(data.size(-1)))
    tensor = CategoricalTensor(data, categories)

    expected = torch.tensor([[False, True, False], [True, False, False]])
    assert torch.isnan(tensor).equal(expected)
    assert tensor.isnan().equal(expected)
    assert torch.isfinite(tensor).equal(~expected)
    assert tensor.isfinite().equal(~expected)
