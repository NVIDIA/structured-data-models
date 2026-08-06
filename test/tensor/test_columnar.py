import copy
import io
from collections.abc import Callable

import pyarrow as pa
import pytest
import torch

from sdm import (
    CategoricalTensor,
    ColumnarTensor,
    NullableIntTensor,
    StringTensor,
)
from sdm.testing import onlyCUDA


def test_init() -> None:
    value1 = torch.arange(6).view(2, 3)
    value2 = torch.randn(2, 3)

    tensor = ColumnarTensor((value1, value2))
    assert tensor.size() == (2, 3, 2)
    assert tensor.device == value1.device == value2.device
    assert repr(tensor) == "ColumnarTensor(size=(2, 3, 2))"

    with pytest.raises(ValueError, match="to have size"):
        ColumnarTensor((torch.ones(2),), size=(3,))

    with pytest.raises(ValueError, match="to have size"):
        ColumnarTensor((torch.ones(2), torch.ones(3)))

    with pytest.raises(TypeError, match="single column tensor"):
        ColumnarTensor(
            (
                CategoricalTensor(
                    code=torch.randint(0, 2, (2, 1)),
                    categories=(torch.arange(2),),
                ),
            )
        )


def test_empty() -> None:
    tensor = ColumnarTensor((), size=(2, 3))
    assert tensor.size() == (2, 3, 0)
    assert tensor.stride() == torch.empty(2, 3, 0).stride()

    with pytest.raises(ValueError, match="zero columnar data"):
        ColumnarTensor(())

    with pytest.raises(ValueError, match="to be non-empty"):
        ColumnarTensor((), size=())


def test_empty_clone_preserves_device() -> None:
    tensor = ColumnarTensor((), size=(2,), device="meta")

    out = tensor.clone()

    assert out.device == tensor.device


def test_tensor_flatten_round_trip() -> None:
    tensor = ColumnarTensor(
        (
            torch.arange(6).view(2, 3),
            StringTensor.from_list([["a", "b", "c"], ["d", "e", "f"]]),
        )
    )
    names, context = tensor.__tensor_flatten__()

    out = ColumnarTensor.__tensor_unflatten__(
        {name: getattr(tensor, name) for name in names},
        context,
        tensor.size(),
        tensor.stride(),
    )

    assert type(out) is type(tensor)
    assert out.size() == tensor.size()
    assert out.stride() == tensor.stride()
    assert out.tolist() == tensor.tolist()

    view = tensor[:, 1:]
    assert isinstance(view, ColumnarTensor)
    names, context = view.__tensor_flatten__()
    out = ColumnarTensor.__tensor_unflatten__(
        {name: getattr(view, name) for name in names},
        context,
        view.size(),
        view.stride(),
    )
    assert out.storage_offset() == view.storage_offset()
    assert out.tolist() == view.tolist()


def test_compile() -> None:
    tensor = ColumnarTensor(
        (
            torch.arange(6).view(2, 3),
            torch.arange(10, 16).view(2, 3),
        )
    )
    compiled = torch.compile(
        lambda value: value.view(6, 2),
        fullgraph=True,
        backend="eager",
    )

    for _ in range(2):
        out = compiled(tensor)
        assert isinstance(out, ColumnarTensor)
        assert out.tolist() == tensor.view(6, 2).tolist()
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
        assert isinstance(out, ColumnarTensor)
        assert out.size() == (1, 2, 3, 2)
        assert out.tolist() == [tensor.tolist()]
        assert torch._C._is_alias_of(  # ty: ignore[unresolved-attribute]
            out,
            tensor,
        )

    compiled_slice = torch.compile(
        lambda value: value[:, 1:],
        fullgraph=True,
        backend="aot_eager",
    )
    eager = tensor[:, 1:]
    out = compiled_slice(tensor)
    assert isinstance(eager, ColumnarTensor)
    assert isinstance(out, ColumnarTensor)
    assert out.tolist() == eager.tolist()
    assert out.stride() == eager.stride()
    assert out.storage_offset() == eager.storage_offset()

    compiled_column_slice = torch.compile(
        lambda value: value[..., 1:],
        fullgraph=True,
        backend="aot_eager",
    )
    eager = tensor[..., 1:]
    out = compiled_column_slice(tensor)
    assert isinstance(out, ColumnarTensor)
    assert out.tolist() == eager.tolist()
    assert out.stride() == eager.stride() == (6, 2, 1)
    assert out.storage_offset() == eager.storage_offset() == 1

    compiled_column_split = torch.compile(
        lambda value: value.split(1, dim=-1),
        fullgraph=True,
        backend="aot_eager",
    )
    eager_parts = tensor.split(1, dim=-1)
    out_parts = compiled_column_split(tensor)
    assert [part.tolist() for part in out_parts] == [
        part.tolist() for part in eager_parts
    ]
    assert [part.storage_offset() for part in out_parts] == [0, 1]
    assert all(
        torch._C._is_alias_of(tensor, part)  # ty: ignore[unresolved-attribute]
        for part in out_parts
    )

    first = torch.arange(6).view(2, 3)
    second = torch.arange(10, 16).view(2, 3)
    construct = torch.compile(
        lambda left, right: ColumnarTensor((left, right)),
        fullgraph=True,
        backend="eager",
    )

    for _ in range(2):
        out = construct(first, second)
        assert isinstance(out, ColumnarTensor)
        assert out.tolist() == tensor.tolist()


@pytest.mark.parametrize(
    "operation",
    [
        lambda value: value[-1],
        lambda value: value[value.size(0) - 2 :],
        lambda value: torch.narrow(value, 0, value.size(0) - 2, 2),
    ],
    ids=("select", "slice", "narrow"),
)
def test_compile_dynamic_view_offset(
    operation: Callable[[ColumnarTensor], torch.Tensor],
) -> None:
    compiled = torch.compile(
        operation,
        fullgraph=True,
        dynamic=True,
        backend="inductor",
    )

    for num_rows in (4, 6):
        tensor = ColumnarTensor(
            (
                torch.arange(num_rows),
                torch.arange(100, 100 + num_rows),
            )
        )
        out = compiled(tensor)
        expected = operation(tensor)
        assert isinstance(out, ColumnarTensor)
        assert isinstance(expected, ColumnarTensor)
        assert out.tolist() == expected.tolist()
        assert out.storage_offset() == expected.storage_offset()
        assert torch._C._is_alias_of(  # ty: ignore[unresolved-attribute]
            out,
            tensor,
        )
        for out_column, expected_column, input_column in zip(
            out._columns,
            expected._columns,
            tensor._columns,
        ):
            assert out_column.storage_offset() == (
                expected_column.storage_offset()
            )
            assert torch._C._is_alias_of(  # ty: ignore[unresolved-attribute]
                out_column,
                input_column,
            )


@pytest.mark.parametrize(
    "dynamic",
    [
        False,
        pytest.param(
            True,
            marks=pytest.mark.skipif(
                tuple(
                    int(part)
                    for part in torch.__version__.split("+", maxsplit=1)[
                        0
                    ].split(".")[:2]
                )
                < (2, 10),
                reason=(
                    "PyTorch before 2.10 cannot compile dynamic "
                    "multidimensional ColumnarTensor views"
                ),
            ),
        ),
    ],
)
def test_compile_flattened_subview(dynamic: bool) -> None:
    compiled = torch.compile(
        lambda value: value.view(-1, 2)[1:],
        fullgraph=True,
        dynamic=dynamic,
        backend="aot_eager",
    )
    layouts = (
        (
            torch.arange(6).view(2, 3),
            torch.arange(100, 106).view(2, 3),
        ),
        (
            torch.as_strided(torch.arange(12), (2, 3), (6, 2)),
            torch.as_strided(torch.arange(100, 112), (2, 3), (6, 2)),
        ),
    )

    for columns in layouts:
        tensor = ColumnarTensor(columns)
        out = compiled(tensor)
        expected = tensor.view(-1, 2)[1:]
        assert out.tolist() == expected.tolist()
        assert out.storage_offset() == expected.storage_offset()


def test_compile_nullable_column_view() -> None:
    nullable = NullableIntTensor(
        data=torch.arange(6).view(2, 3),
        valid=torch.tensor([[True, False, True], [False, True, True]]),
    )
    tensor = ColumnarTensor((nullable,))
    compiled = torch.compile(
        lambda value: value[-1],
        fullgraph=True,
        backend="aot_eager",
    )

    out = compiled(tensor)
    expected = tensor[-1]
    assert out.tolist() == expected.tolist()
    assert out.storage_offset() == expected.storage_offset()


def test_from_arrow() -> None:
    tensor = ColumnarTensor.from_arrow(pa.array([1, 2, 3]))
    assert tensor.size() == (3, 1)
    assert isinstance(tensor._columns[0], torch.Tensor)
    assert not isinstance(tensor._columns[0], StringTensor)
    assert tensor.tolist() == [[1], [2], [3]]

    tensor = ColumnarTensor.from_arrow(pa.array(["a", "bb", ""]))
    assert tensor.size() == (3, 1)
    assert isinstance(tensor._columns[0], StringTensor)
    assert tensor.tolist() == [["a"], ["bb"], [""]]

    tensor = ColumnarTensor.from_arrow(pa.array([1, None, 3]))
    assert isinstance(tensor._columns[0], NullableIntTensor)
    assert tensor.tolist() == [[1], [None], [3]]
    assert tensor.to_arrow().to_pydict() == {"0": [1, None, 3]}


def test_from_arrow_chunked_string() -> None:
    tensor = ColumnarTensor.from_arrow(
        pa.chunked_array([pa.array(["a", "b"]), pa.array(["c"])]),
    )

    assert tensor.to_arrow().column(0).type == pa.large_string()
    assert tensor.to_arrow().to_pydict() == {"0": ["a", "b", "c"]}


@onlyCUDA
def test_from_arrow_cuda() -> None:
    tensor = ColumnarTensor.from_arrow(pa.array([1, 2, 3]), device="cuda")
    assert tensor.is_cuda
    assert tensor[:, 0].equal(torch.tensor([1, 2, 3], device=tensor.device))


@onlyCUDA
def test_from_cudf() -> None:
    cudf = pytest.importorskip("cudf")

    tensor = ColumnarTensor.from_cudf(
        cudf.Series([1, 2, 3], dtype="int64"),
    )
    assert tensor.size() == (3, 1)
    assert tensor.is_cuda
    assert tensor[:, 0].equal(torch.tensor([1, 2, 3], device=tensor.device))

    tensor = ColumnarTensor.from_cudf(
        cudf.Series([1.5, None, 3.5], dtype="float32"),
    )
    assert tensor.size() == (3, 1)
    assert tensor.is_cuda
    assert tensor[:, 0].is_floating_point()
    assert tensor[:, 0].allclose(
        torch.tensor([1.5, float("nan"), 3.5], device=tensor.device),
        equal_nan=True,
    )

    tensor = ColumnarTensor.from_cudf(
        cudf.Series(["a", "bb", ""]),
    )
    assert tensor.size() == (3, 1)
    assert tensor.is_cuda
    assert isinstance(tensor._columns[0], StringTensor)
    assert tensor.tolist() == [["a"], ["bb"], [""]]

    tensor = ColumnarTensor.from_cudf(
        cudf.Series([1, None, 3], dtype="int64"),
    )
    assert isinstance(tensor._columns[0], NullableIntTensor)
    assert tensor.tolist() == [[1], [None], [3]]
    assert tensor.to_cudf().to_arrow().to_pydict() == {"0": [1, None, 3]}


def test_to_arrow() -> None:
    column1 = torch.arange(6).view(2, 3)
    column2 = StringTensor.from_list([["a", "b", "c"], ["d", "e", "f"]])
    tensor = ColumnarTensor((column1, column2))

    table = tensor.to_arrow()
    assert table.column_names == ["0", "1"]
    assert table.to_pydict() == {
        "0": [0, 1, 2, 3, 4, 5],
        "1": ["a", "b", "c", "d", "e", "f"],
    }


@onlyCUDA
def test_to_cudf() -> None:
    pytest.importorskip("cudf")

    column1 = torch.arange(6, device="cuda").view(2, 3)
    column2 = StringTensor.from_list(
        [["a", "b", "c"], ["d", "e", "f"]], device="cuda"
    )
    tensor = ColumnarTensor((column1, column2))

    df = tensor.to_cudf()
    assert df.columns.tolist() == ["0", "1"]
    assert df.to_arrow().to_pydict() == {
        "0": [0, 1, 2, 3, 4, 5],
        "1": ["a", "b", "c", "d", "e", "f"],
    }


def test_save_load() -> None:
    tensor = ColumnarTensor(
        (torch.arange(6).view(2, 3), torch.arange(10, 16).view(2, 3))
    )[:, 1:]

    buffer = io.BytesIO()
    torch.save(tensor, buffer)
    buffer.seek(0)
    out = torch.load(buffer, weights_only=False)

    assert isinstance(out, ColumnarTensor)
    assert out.size() == tensor.size()
    assert out.stride() == tensor.stride()
    assert out.storage_offset() == tensor.storage_offset()
    assert out.tolist() == tensor.tolist()

    base = ColumnarTensor(
        (torch.arange(6).view(2, 3), torch.arange(10, 16).view(2, 3))
    )
    view = base[:, 1:]
    buffer = io.BytesIO()
    torch.save((base, view), buffer)
    buffer.seek(0)
    loaded_base, loaded_view = torch.load(buffer, weights_only=False)

    assert torch._C._is_alias_of(  # ty: ignore[unresolved-attribute]
        loaded_base,
        loaded_view,
    )
    assert all(
        torch._C._is_alias_of(  # ty: ignore[unresolved-attribute]
            base_column,
            view_column,
        )
        for base_column, view_column in zip(
            loaded_base._columns,
            loaded_view._columns,
        )
    )


def test_deepcopy_view() -> None:
    tensor = ColumnarTensor(
        (torch.arange(6).view(2, 3), torch.arange(10, 16).view(2, 3))
    )[:, 1:]

    out = copy.deepcopy(tensor)

    assert isinstance(out, ColumnarTensor)
    assert out.size() == tensor.size()
    assert out.stride() == tensor.stride()
    assert out.storage_offset() == tensor.storage_offset()
    assert out.tolist() == tensor.tolist()


def test_materialized_column_slice_has_independent_topology() -> None:
    tensor = ColumnarTensor(
        tuple(
            torch.arange(start, start + 6).view(2, 3) for start in (0, 10, 20)
        )
    )
    view = tensor[..., 1:]
    assert isinstance(view, ColumnarTensor)
    identity_as_strided = torch.compile(
        lambda value: torch.as_strided(
            value,
            (2, 3, 2),
            (6, 2, 1),
            0,
        ),
        fullgraph=True,
        backend="eager",
    )

    for out in (view.clone(), view.contiguous()):
        assert isinstance(out, ColumnarTensor)
        identity = identity_as_strided(out)
        assert isinstance(identity, ColumnarTensor)
        assert identity.tolist() == view.tolist()

        out.unbind(-1)[0].fill_(-1)
        assert tensor.unbind(-1)[1].equal(torch.arange(10, 16).view(2, 3))


def test_to_copy() -> None:
    column1 = torch.arange(12).view(2, 3, 2)[..., 0]
    column2 = torch.randn(2, 3, 2)[..., 1]
    tensor = ColumnarTensor((column1, column2))

    assert not tensor.is_contiguous()

    out = tensor.contiguous()
    assert isinstance(out, ColumnarTensor)
    assert out.size() == tensor.size()
    assert out.is_contiguous()
    assert out.tolist() == tensor.tolist()

    out = tensor.clone()
    assert isinstance(out, ColumnarTensor)
    assert out.size() == tensor.size()
    assert out.tolist() == tensor.tolist()

    out = tensor.to("cpu")
    assert isinstance(out, ColumnarTensor)
    assert out.is_cpu
    assert out.tolist() == tensor.tolist()

    with pytest.raises(TypeError, match="convert"):
        tensor.to(torch.float32)

    tensor = ColumnarTensor(
        (
            torch.arange(6).view(2, 3),
            torch.arange(10, 16).view(2, 3),
        )
    ).transpose(0, 1)
    out = tensor.clone(memory_format=torch.preserve_format)
    assert out.stride() == tensor.stride() == (2, 6, 1)
    assert out.tolist() == tensor.tolist()


@onlyCUDA
def test_to_cuda() -> None:
    tensor = ColumnarTensor(
        (
            torch.arange(3),
            StringTensor.from_list(["a", "bb", "c"]),
        )
    )

    out = tensor.to("cuda")
    assert isinstance(out, ColumnarTensor)
    assert out.device.type == "cuda"
    assert out.tolist() == tensor.tolist()


def test_view_ops() -> None:
    tensor = ColumnarTensor(
        (
            torch.arange(6).view(2, 3),
            torch.arange(10, 16).view(2, 3),
        )
    )

    out = tensor.view(6, 2)
    assert isinstance(out, ColumnarTensor)
    assert out.size() == (6, 2)
    assert out.tolist() == [
        [0, 10],
        [1, 11],
        [2, 12],
        [3, 13],
        [4, 14],
        [5, 15],
    ]
    assert torch._C._is_alias_of(  # ty: ignore[unresolved-attribute]
        out,
        tensor,
    )

    with pytest.raises(RuntimeError, match="Can't reshape"):
        _ = tensor.view(-1)

    out = tensor.unsqueeze(0)
    assert isinstance(out, ColumnarTensor)
    assert out.size() == (1, 2, 3, 2)

    out = tensor.unsqueeze(0).squeeze(0)
    assert isinstance(out, ColumnarTensor)
    assert out.size() == tensor.size()

    with pytest.raises(RuntimeError, match="unsqueeze"):
        _ = tensor.unsqueeze(-1)

    out = tensor.unsqueeze(1).expand(-1, 4, 3, -1)
    assert isinstance(out, ColumnarTensor)
    assert out.size() == (2, 4, 3, 2)

    out = tensor.transpose(0, 1)
    assert isinstance(out, ColumnarTensor)
    assert out.size() == (3, 2, 2)

    out = tensor.permute(1, 0, 2)
    assert isinstance(out, ColumnarTensor)
    assert out.size() == (3, 2, 2)

    with pytest.raises(RuntimeError, match="column dimension"):
        _ = tensor.permute(2, 0, 1)


def test_slicing_ops() -> None:
    tensor = ColumnarTensor(
        (
            torch.arange(24).view(2, 3, 4),
            torch.arange(100, 124).view(2, 3, 4),
        )
    )

    out = tensor.select(0, 1)
    assert isinstance(out, ColumnarTensor)
    assert out.size() == (3, 4, 2)

    out = tensor.select(-1, 1)
    assert not isinstance(out, ColumnarTensor)
    assert out.equal(tensor._columns[1])

    out = tensor[:, 1:]
    assert isinstance(out, ColumnarTensor)
    assert out.size() == (2, 2, 4, 2)

    out = tensor[..., 1:]
    assert isinstance(out, ColumnarTensor)
    assert out.size() == (2, 3, 4, 1)
    assert out._columns == (tensor._columns[1],)
    assert not out.is_contiguous()
    assert out.contiguous().is_contiguous()

    out = tensor.narrow(1, 1, 1)
    assert isinstance(out, ColumnarTensor)
    assert out.size() == (2, 1, 4, 2)

    rows = tensor.unbind(0)
    assert len(rows) == 2
    assert all(isinstance(row, ColumnarTensor) for row in rows)
    assert rows[0].size() == (3, 4, 2)
    assert all(
        torch._C._is_alias_of(tensor, row)  # ty: ignore[unresolved-attribute]
        for row in rows
    )

    columns = tensor.unbind(-1)
    assert len(columns) == 2
    assert columns[0].equal(tensor._columns[0])
    assert columns[1].equal(tensor._columns[1])

    chunks = tensor.split(1, dim=1)
    assert len(chunks) == 3
    assert all(isinstance(chunk, ColumnarTensor) for chunk in chunks)
    assert chunks[0].size() == (2, 1, 4, 2)
    assert all(
        torch._C._is_alias_of(tensor, chunk)  # ty: ignore[unresolved-attribute]
        for chunk in chunks
    )

    chunks = tensor.split(1, dim=-1)
    assert len(chunks) == 2
    assert chunks[0].size() == (2, 3, 4, 1)
    assert chunks[0]._columns == (tensor._columns[0],)


def test_view_metadata_is_independent_of_column_layout() -> None:
    values = torch.arange(48).view(2, 3, 8)
    tensor = ColumnarTensor((values[..., 0], values[..., 1]))

    out = tensor[:, 1:]

    assert out.stride() == (6, 2, 1)
    assert out.storage_offset() == 2
    assert out.tolist() == [
        [[8, 9], [16, 17]],
        [[32, 33], [40, 41]],
    ]

    compiled = torch.compile(
        lambda value: value[:, 1:],
        fullgraph=True,
        backend="aot_eager",
    )
    compiled_out = compiled(tensor)
    assert compiled_out.stride() == out.stride()
    assert compiled_out.storage_offset() == out.storage_offset()
    assert compiled_out.tolist() == out.tolist()

    compiled_view = torch.compile(
        lambda value: value.unsqueeze(0),
        fullgraph=True,
        backend="aot_eager",
    )
    compiled_out = compiled_view(out)
    assert compiled_out.tolist() == [out.tolist()]
    assert torch._C._is_alias_of(  # ty: ignore[unresolved-attribute]
        out,
        compiled_out,
    )


def test_as_strided_crosses_rows_between_columns() -> None:
    columns = (
        torch.tensor([0, 1, 2]),
        torch.tensor([10, 11, 12]),
        torch.tensor([20, 21, 22]),
    )
    tensor = ColumnarTensor(columns)
    dense = torch.stack(columns, dim=-1)

    for stride in ((3, 4), (3, 3)):
        out = torch.as_strided(tensor, (1, 3), stride, 0)
        expected = torch.as_strided(dense, (1, 3), stride, 0)
        assert out.tolist() == expected.tolist()


@pytest.mark.skipif(
    tuple(
        int(part)
        for part in torch.__version__.split("+", maxsplit=1)[0].split(".")[:2]
    )
    < (2, 10),
    reason=(
        "PyTorch before 2.10 cannot regenerate dynamic multidimensional "
        "ColumnarTensor aliases"
    ),
)
def test_compile_as_strided_crosses_rows_between_columns() -> None:
    compiled = torch.compile(
        lambda value: torch.as_strided(
            value,
            (value.size(0), 2, 3),
            value.stride(),
            0,
        ),
        fullgraph=True,
        dynamic=True,
        backend="aot_eager",
    )
    for num_rows in (4, 6):
        dynamic_columns = tuple(
            torch.arange(start, start + num_rows * 3).view(num_rows, 3)
            for start in (0, 100, 200)
        )
        dynamic_tensor = ColumnarTensor(dynamic_columns)
        dynamic_dense = torch.stack(dynamic_columns, dim=-1)
        expected = torch.as_strided(
            dynamic_dense,
            (num_rows, 2, 3),
            dynamic_dense.stride(),
            0,
        )
        assert compiled(dynamic_tensor).tolist() == expected.tolist()


def test_as_strided_rejects_non_affine_column_layout() -> None:
    column = torch.as_strided(
        torch.arange(20),
        (2, 3),
        (10, 1),
    )
    tensor = ColumnarTensor((column,))

    with pytest.raises(RuntimeError, match="cannot be replayed"):
        torch.as_strided(tensor, (2, 3, 1), (2, 1, 1))


def test_empty_view_metadata() -> None:
    tensor = ColumnarTensor((), size=(2, 3))

    out = tensor[:, 1:]

    assert out.size() == (2, 2, 0)
    assert out.stride() == (3, 1, 1)
    assert out.storage_offset() == 1


def test_index_ops() -> None:
    tensor = ColumnarTensor(
        (
            torch.arange(24).view(2, 3, 4),
            torch.arange(100, 124).view(2, 3, 4),
        )
    )

    out = tensor.index_select(1, torch.tensor([2, 0]))
    assert isinstance(out, ColumnarTensor)
    assert out.size() == (2, 2, 4, 2)

    out = tensor.index_select(-1, torch.tensor([1, 0]))
    assert isinstance(out, ColumnarTensor)
    assert out.size() == tensor.size()
    assert out._columns[0].equal(tensor._columns[1])
    assert out._columns[1].equal(tensor._columns[0])

    out = tensor[:, torch.tensor([2, 0])]
    assert isinstance(out, ColumnarTensor)
    assert out.size() == (2, 2, 4, 2)

    out = tensor[..., torch.tensor([1, 0])]
    assert isinstance(out, ColumnarTensor)
    assert out.size() == tensor.size()
    assert out._columns[0].equal(tensor._columns[1])
    assert out._columns[1].equal(tensor._columns[0])

    out = tensor[..., torch.tensor([True, False])]
    assert isinstance(out, ColumnarTensor)
    assert out.size() == (2, 3, 4, 1)
    assert out._columns[0].equal(tensor._columns[0])

    mask = torch.tensor([[True, False, True], [False, True, False]])
    out = tensor[mask]
    assert isinstance(out, ColumnarTensor)
    assert out.size() == (3, 4, 2)

    with pytest.raises(RuntimeError, match="column dimension"):
        _ = tensor[torch.tensor([0]), :, :, torch.tensor([1])]


def test_cat_stack() -> None:
    tensor1 = ColumnarTensor(
        (
            torch.arange(6).view(2, 3),
            torch.arange(10, 16).view(2, 3),
        )
    )
    tensor2 = ColumnarTensor(
        (
            torch.arange(20, 26).view(2, 3),
            torch.arange(30, 36).view(2, 3),
        )
    )

    out = torch.cat([tensor1, tensor2], dim=0)
    assert isinstance(out, ColumnarTensor)
    assert out.size() == (4, 3, 2)
    assert out._columns[0].equal(
        torch.cat([tensor1._columns[0], tensor2._columns[0]])
    )

    out = torch.cat([tensor1, tensor2], dim=-1)
    assert isinstance(out, ColumnarTensor)
    assert out.size() == (2, 3, 4)
    assert out._columns == (*tensor1._columns, *tensor2._columns)

    out = torch.stack([tensor1, tensor2], dim=0)
    assert isinstance(out, ColumnarTensor)
    assert out.size() == (2, 2, 3, 2)

    with pytest.raises(RuntimeError, match="stack after"):
        _ = torch.stack([tensor1, tensor2], dim=-1)


def test_tolist() -> None:
    tensor = ColumnarTensor(
        (
            torch.tensor([[1, 2], [3, 4]]),
            torch.tensor([[10, 20], [30, 40]]),
        )
    )

    assert tensor.tolist() == [
        [[1, 10], [2, 20]],
        [[3, 30], [4, 40]],
    ]


def test_pin_memory() -> None:
    tensor = ColumnarTensor(
        (
            torch.randn(2, 3),
            torch.arange(6).view(2, 3),
        )
    )

    assert not tensor.is_pinned()


@onlyCUDA
def test_pin_memory_cuda() -> None:
    tensor = ColumnarTensor(
        (
            torch.randn(2, 3),
            torch.arange(6).view(2, 3),
        )
    )

    assert tensor.pin_memory().is_pinned()


def test_share_memory() -> None:
    tensor = ColumnarTensor(
        (
            torch.randn(2, 3),
            torch.arange(6).view(2, 3),
        )
    )

    assert not tensor.is_shared()
    try:
        tensor.share_memory_()
        assert tensor.is_shared()
    except RuntimeError:
        pass
