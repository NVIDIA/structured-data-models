from typing import cast

import pyarrow as pa
import pytest
import torch
from torch import Tensor
from torch._subclasses.fake_tensor import FakeTensorMode

from sdm import VarLenTensor
from sdm.testing import onlyCUDA


def test_dtype_conversion() -> None:
    tensor = VarLenTensor.from_tensor(torch.arange(4).view(2, 2))
    assert repr(tensor) == "VarLenTensor(size=(2, 2), dtype=torch.int64)"

    out = tensor.to(torch.float64)
    assert isinstance(out, VarLenTensor)
    assert out.dtype == torch.float64
    assert out._data.dtype == torch.float64
    assert out._offset.dtype == torch.int64


def test_autograd() -> None:
    data = torch.randn(4, requires_grad=True)
    tensor = VarLenTensor.from_tensor(data)
    assert tensor.requires_grad
    assert repr(tensor) == (
        "VarLenTensor(size=(4,), dtype=torch.float32, requires_grad=True)"
    )

    out = tensor.clone()
    assert isinstance(out, VarLenTensor)
    assert repr(out) == (
        "VarLenTensor(size=(4,), dtype=torch.float32, "
        "grad_fn=<ToCopyBackward0>)"
    )

    out._data.sum().backward()
    assert data.grad is not None
    assert data.grad.equal(torch.ones_like(data))

    tensor = tensor.detach()
    assert not tensor.requires_grad
    tensor.requires_grad_(True)
    assert tensor.requires_grad


def test_offset_dtype() -> None:
    tensor = VarLenTensor.from_tensor(
        torch.arange(4).view(2, 2),
        offset_dtype=torch.int32,
    )
    out = tensor.masked_select(torch.tensor([[True, False], [False, True]]))
    assert isinstance(out, VarLenTensor)
    assert out._offset.dtype == torch.int32

    out = torch.cat(
        [
            VarLenTensor(
                data=torch.tensor([0, 1, 2]),
                offset=torch.tensor([0, 1, 3], dtype=torch.int32),
                valid=None,
                size=(1, 2),
            ),
            VarLenTensor(
                data=torch.tensor([3, 4]),
                offset=torch.tensor([0, 1, 2], dtype=torch.int32),
                valid=None,
                size=(1, 2),
            ),
        ]
    )
    assert isinstance(out, VarLenTensor)
    assert out._offset.dtype == torch.int32

    out = torch.cat(
        [
            VarLenTensor(
                data=torch.tensor([0, 1, 2]),
                offset=torch.tensor([0, 1, 3], dtype=torch.int32),
                valid=None,
                size=(1, 2),
            ),
            VarLenTensor(
                data=torch.tensor([3, 4]),
                offset=torch.tensor([0, 1, 2], dtype=torch.int64),
                valid=None,
                size=(1, 2),
            ),
        ]
    )
    assert isinstance(out, VarLenTensor)
    assert out._offset.dtype == torch.int64


def test_fake_data_validates_offset_dtype_capacity() -> None:
    with FakeTensorMode():
        data = torch.empty(torch.iinfo(torch.int32).max + 1)
        offset = torch.zeros(2, dtype=torch.int32)

        with pytest.raises(
            RuntimeError,
            match=r"'torch\.int32' can only represent 2147483647 elements",
        ):
            VarLenTensor(
                data=data,
                offset=offset,
                valid=None,
                size=(1,),
            )


def test_arrow() -> None:
    tensor = VarLenTensor.from_arrow(
        pa.array([[1, 2], [], [3]], type=pa.list_(pa.int64())),
    )
    assert tensor.size() == (3,)
    assert tensor.dtype == torch.int64
    assert tensor._data.equal(torch.tensor([1, 2, 3]))
    assert tensor._offset.equal(torch.tensor([0, 2, 2, 3]))
    array = tensor.to_arrow()
    assert array.type == pa.list_(pa.int64())
    assert array.to_pylist() == [[1, 2], [], [3]]

    tensor = VarLenTensor.from_arrow(
        pa.array([[1.0, 2.0], [3.0]], type=pa.large_list(pa.float32()))[1:],
    )
    assert tensor.storage_offset() == 1
    assert tensor.dtype == torch.float32
    assert tensor._data.equal(torch.tensor([1.0, 2.0, 3.0]))
    assert tensor._offset.equal(torch.tensor([0, 2, 3]))
    array = tensor.to_arrow()
    assert array.type == pa.large_list(pa.float32())
    assert array.to_pylist() == [[3.0]]

    values = pa.array([0, 1, 2, 3, 4], type=pa.int64())[2:]
    tensor = VarLenTensor.from_arrow(
        pa.ListArray.from_arrays(
            pa.array([0, 1, 3], type=pa.int32()),
            values,
        ),
    )
    assert tensor._data.equal(torch.tensor([2, 3, 4]))
    assert tensor._offset.equal(torch.tensor([0, 1, 3]))
    assert tensor.tolist() == [[2], [3, 4]]

    values = pa.array([0, 1, 2, 3, 4], type=pa.int64()).slice(2, 2)
    tensor = VarLenTensor.from_arrow(
        pa.ListArray.from_arrays(
            pa.array([0, 1, 2], type=pa.int32()),
            values,
        ),
    )
    assert tensor._data.equal(torch.tensor([2, 3]))
    assert tensor._offset.equal(torch.tensor([0, 1, 2]))
    assert tensor.tolist() == [[2], [3]]
    array = tensor.to_arrow()
    assert array.values.to_pylist() == [2, 3]
    assert array.to_pylist() == [[2], [3]]

    tensor = VarLenTensor.from_arrow(
        pa.array([[1, 2], None, [3]], type=pa.list_(pa.int64())),
    )
    assert tensor.valid is not None
    assert tensor.valid.equal(torch.tensor([True, False, True]))
    assert tensor.to_arrow().to_pylist() == [[1, 2], None, [3]]
    assert tensor.tolist() == [[1, 2], None, [3]]


def test_list() -> None:
    data = [
        [[1, 2], [3, 4, 5]],
        [[], [6]],
        [[7, 8], None],
    ]
    tensor = VarLenTensor.from_list(data)
    assert tensor.size() == (3, 2)
    assert tensor.dtype == torch.int64
    assert tensor._data.equal(torch.tensor([1, 2, 3, 4, 5, 6, 7, 8]))
    assert tensor._offset.equal(torch.tensor([0, 2, 5, 5, 6, 8, 8]))
    assert tensor.tolist() == data
    assert tensor[0, 0].item() == [1, 2]
    assert tensor[1, 0].item() == []
    assert tensor[2, 1].item() is None


def test_tensor_flatten_round_trip() -> None:
    tensor = cast(
        VarLenTensor,
        VarLenTensor.from_list([[1, 2], None, [3]])[1:],
    )
    names, context = tensor.__tensor_flatten__()

    out = VarLenTensor.__tensor_unflatten__(
        {name: getattr(tensor, name) for name in names},
        context,
        tensor.size(),
        tensor.stride(),
    )

    assert type(out) is type(tensor)
    assert out.size() == tensor.size()
    assert out.stride() == tensor.stride()
    assert out.storage_offset() == tensor.storage_offset()
    assert out.tolist() == tensor.tolist()


def test_empty_contiguous_stride() -> None:
    tensor = VarLenTensor(
        data=torch.empty(0),
        offset=torch.zeros(1, dtype=torch.int64),
        valid=None,
        size=(2, 0, 4),
    )

    assert tensor.stride() == torch.empty(2, 0, 4).stride()


def test_empty_view_storage_offset() -> None:
    tensor = VarLenTensor(
        data=torch.empty(0),
        offset=torch.zeros(1, dtype=torch.int64),
        valid=torch.empty(0, dtype=torch.bool),
        size=(2, 3, 4, 0),
    )

    out = cast(VarLenTensor, tensor[:, :, 1:3])

    assert out.size() == (2, 3, 2, 0)
    assert out.stride() == (12, 4, 1, 1)
    assert out.storage_offset() == 1
    assert torch._C._is_alias_of(  # ty: ignore[unresolved-attribute]
        out,
        tensor,
    )
    assert out.valid is not None
    assert out.valid.size() == out.size()

    data, offset = out.data_offset
    assert data.numel() == 0
    assert offset.equal(torch.zeros(1, dtype=torch.int64))
    assert out.to_arrow().to_pylist() == []

    cloned = out.clone()
    assert isinstance(cloned, VarLenTensor)
    assert cloned.size() == out.size()
    assert cloned.stride() == out.stride()
    assert cloned.storage_offset() == 0
    assert cloned._offset.equal(torch.zeros(1, dtype=torch.int64))
    assert cloned._valid is not None
    assert cloned._valid.numel() == 0

    compiled_clone = torch.compile(
        lambda value: value.clone(),
        fullgraph=True,
        backend="aot_eager",
    )
    compiled = compiled_clone(out)
    assert compiled.size() == out.size()
    assert compiled.stride() == out.stride()
    assert compiled.storage_offset() == 0
    assert compiled._offset.equal(torch.zeros(1, dtype=torch.int64))


def test_compile() -> None:
    tensor = VarLenTensor.from_list([[1], [2, 3]])
    compiled = torch.compile(
        lambda value: value.view(-1),
        fullgraph=True,
        backend="eager",
    )

    for _ in range(2):
        out = compiled(tensor)
        assert isinstance(out, VarLenTensor)
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
        assert isinstance(out, VarLenTensor)
        assert out.size() == (1, 2)
        assert out.tolist() == [tensor.tolist()]
        assert torch._C._is_alias_of(  # ty: ignore[unresolved-attribute]
            out,
            tensor,
        )

    data = torch.tensor([1, 2, 3])
    offset = torch.tensor([0, 1, 3])
    construct = torch.compile(
        lambda values, offsets: VarLenTensor(
            data=values,
            offset=offsets,
            valid=None,
            size=(2,),
        ),
        fullgraph=True,
        backend="eager",
    )

    for _ in range(2):
        out = construct(data, offset)
        assert isinstance(out, VarLenTensor)
        assert out.tolist() == [[1], [2, 3]]

    clone = torch.compile(
        lambda value: value.clone(),
        fullgraph=True,
        backend="aot_eager",
    )
    for _ in range(2):
        out = clone(tensor)
        assert isinstance(out, VarLenTensor)
        assert out.tolist() == tensor.tolist()
        assert not torch._C._is_alias_of(  # ty: ignore[unresolved-attribute]
            out,
            tensor,
        )

    sliced = cast(VarLenTensor, tensor[1:])
    out = clone(sliced)
    assert out.tolist() == sliced.tolist()
    assert out.storage_offset() == 0

    convert = torch.compile(
        lambda value: value.to(torch.float64),
        fullgraph=True,
        backend="aot_eager",
    )
    out = convert(tensor)
    assert isinstance(out, VarLenTensor)
    assert out.dtype == torch.float64
    assert out.tolist() == tensor.tolist()

    values = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    reduce = torch.compile(
        lambda data, offsets: VarLenTensor(
            data=data + 0,
            offset=offsets,
            valid=None,
            size=(2,),
        )._data.sum(),
        fullgraph=True,
        backend="aot_eager",
    )
    reduce(values, offset).backward()
    assert values.grad is not None
    assert values.grad.equal(torch.ones_like(values))


@pytest.mark.skipif(
    tuple(
        int(part)
        for part in torch.__version__.split("+", maxsplit=1)[0].split(".")[:2]
    )
    < (2, 10),
    reason="PyTorch before 2.10 cannot compile dynamic VarLen compaction",
)
def test_compile_materializes_non_dense_layouts() -> None:
    tensor = VarLenTensor.from_list(
        [[value, value + 1] for value in range(12)]
    ).view(3, 4)
    inputs = (
        cast(VarLenTensor, tensor[:, ::2]),
        cast(VarLenTensor, tensor[:1].expand(3, 4)),
    )
    clone = torch.compile(
        lambda value: value.clone(),
        fullgraph=True,
        backend="aot_eager",
    )
    convert = torch.compile(
        lambda value: value.to(torch.float64),
        fullgraph=True,
        backend="aot_eager",
    )

    for inp in inputs:
        cloned = clone(inp)
        assert isinstance(cloned, VarLenTensor)
        assert cloned.is_contiguous()
        assert cloned.tolist() == inp.tolist()

        converted = convert(inp)
        assert isinstance(converted, VarLenTensor)
        assert converted.is_contiguous()
        assert converted.dtype == torch.float64
        assert converted.tolist() == inp.tolist()


def test_to_copy() -> None:
    data = torch.arange(16)
    offset = torch.arange(data.numel() + 1)

    tensor = VarLenTensor(
        data=data,
        offset=offset,
        valid=None,
        size=(4, 4),
    )
    out = tensor.clone()
    assert isinstance(out, VarLenTensor)
    assert out.size() == tensor.size()
    assert out.stride() == tensor.stride()
    assert out._data.equal(tensor._data)
    assert out._offset.equal(tensor._offset)
    assert out._data.data_ptr() != tensor._data.data_ptr()
    assert out._offset.data_ptr() != tensor._offset.data_ptr()

    assert tensor.to(copy=False) is tensor
    assert tensor.to(torch.int64, copy=False) is tensor

    valid = torch.tensor([True, False, True, True])
    tensor = VarLenTensor(
        data=torch.arange(4),
        offset=torch.arange(5),
        valid=valid,
        size=(4,),
    )
    out = tensor.clone()
    assert isinstance(out, VarLenTensor)
    assert out._data.data_ptr() != tensor._data.data_ptr()
    assert out._offset.data_ptr() != tensor._offset.data_ptr()
    assert out._valid is not None
    assert tensor._valid is not None
    assert out._valid.data_ptr() != tensor._valid.data_ptr()

    tensor = VarLenTensor(
        data=data,
        offset=offset,
        valid=None,
        size=(4, 2),
        stride=(1, 4),
        storage_offset=2,
    )
    out = tensor.clone()
    assert isinstance(out, VarLenTensor)
    assert out.stride() == tensor.stride()
    assert out.storage_offset() == 0
    assert out._data.equal(torch.arange(2, 10))
    assert out._offset.equal(torch.arange(9))
    assert out._data.data_ptr() != tensor._data.data_ptr()
    assert out._offset.data_ptr() != tensor._offset.data_ptr()

    out = tensor.clone(memory_format=torch.contiguous_format)
    assert isinstance(out, VarLenTensor)
    assert out.stride() == (2, 1)
    assert out._data.equal(torch.tensor([2, 6, 3, 7, 4, 8, 5, 9]))
    assert out._offset.equal(torch.arange(9))
    assert out._data.data_ptr() != tensor._data.data_ptr()
    assert out._offset.data_ptr() != tensor._offset.data_ptr()

    tensor = VarLenTensor(
        data=data,
        offset=offset,
        valid=None,
        size=(4, 2),
        stride=(4, 1),
        storage_offset=2,
    )
    out = tensor.clone()
    assert isinstance(out, VarLenTensor)
    assert out.stride() == (2, 1)
    assert out._data.equal(torch.tensor([2, 3, 6, 7, 10, 11, 14, 15]))
    assert out._offset.equal(torch.arange(9))
    assert out._data.data_ptr() != tensor._data.data_ptr()
    assert out._offset.data_ptr() != tensor._offset.data_ptr()

    tensor = VarLenTensor(
        data=data,
        offset=offset,
        valid=None,
        size=(4, 4),
        stride=(0, 1),
    )
    out = tensor.clone()
    assert isinstance(out, VarLenTensor)
    assert out.stride() == (4, 1)
    assert out._data.equal(torch.tensor([0, 1, 2, 3] * 4))
    assert out._offset.equal(torch.arange(17))
    assert out._data.data_ptr() != tensor._data.data_ptr()
    assert out._offset.data_ptr() != tensor._offset.data_ptr()

    tensor = VarLenTensor(
        data=data,
        offset=offset,
        valid=None,
        size=(2,),
        storage_offset=2,
    )
    out = tensor.contiguous()
    assert isinstance(out, VarLenTensor)
    assert out is tensor
    assert out.storage_offset() == 2
    assert out._data.data_ptr() == tensor._data.data_ptr()
    assert out._offset.data_ptr() == tensor._offset.data_ptr()

    out = tensor.clone()
    assert isinstance(out, VarLenTensor)
    assert out.storage_offset() == 0
    assert out._data.equal(torch.tensor([2, 3]))
    assert out._offset.equal(torch.tensor([0, 1, 2]))
    assert out._data.data_ptr() != tensor._data.data_ptr()
    assert out._offset.data_ptr() != tensor._offset.data_ptr()


def test_to_inference_mode() -> None:
    with torch.inference_mode():
        tensor = VarLenTensor(
            data=torch.arange(4, dtype=torch.float32),
            offset=torch.arange(5),
            valid=torch.tensor([True, False, True, True]),
            size=(4,),
        )

        out = tensor.to(torch.float64)
        assert isinstance(out, VarLenTensor)
        assert out.dtype == torch.float64
        assert out._offset.data_ptr() != tensor._offset.data_ptr()
        assert out._valid is not None
        assert tensor._valid is not None
        assert out._valid.data_ptr() == tensor._valid.data_ptr()

        other = torch.empty((), dtype=torch.float64)
        out = tensor.to(other)
        assert isinstance(out, VarLenTensor)
        assert out.dtype == other.dtype

        out = tensor.to(copy=True)
        assert isinstance(out, VarLenTensor)
        assert out._offset.data_ptr() != tensor._offset.data_ptr()
        assert out._valid is not None
        assert tensor._valid is not None
        assert out._valid.data_ptr() != tensor._valid.data_ptr()


@onlyCUDA
def test_to_device_cuda() -> None:
    data = torch.arange(16)
    offset = torch.arange(data.numel() + 1)

    tensor = VarLenTensor(
        data=data,
        offset=offset,
        valid=None,
        size=(4, 4),
    )
    out = tensor.to("cuda")
    assert isinstance(out, VarLenTensor)
    assert out.device.type == "cuda"
    assert out.size() == tensor.size()
    assert out.stride() == tensor.stride()
    assert out._data.device.type == "cuda"
    assert out._offset.device.type == "cuda"
    assert out._data.equal(data.to(out.device))
    assert out._offset.equal(offset.to(out.device))

    back = out.to("cpu")
    assert isinstance(back, VarLenTensor)
    assert back.device.type == "cpu"
    assert back.size() == tensor.size()
    assert back.stride() == tensor.stride()
    assert back._data.equal(data)
    assert back._offset.equal(offset)


def test_data_offset() -> None:
    tensor = VarLenTensor(
        data=torch.arange(8),
        offset=torch.arange(9),
        valid=None,
        size=(2,),
        storage_offset=2,
    )
    data, offset = tensor.data_offset
    assert data.equal(torch.tensor([2, 3]))
    assert offset.equal(torch.tensor([0, 1, 2]))

    tensor = VarLenTensor(
        data=torch.arange(8),
        offset=torch.arange(9),
        valid=None,
        size=(2, 4),
    )
    with pytest.raises(RuntimeError, match="non-contiguous"):
        _ = cast(VarLenTensor, tensor[:, ::2]).data_offset


def test_equal_allclose() -> None:
    tensor = VarLenTensor(
        data=torch.tensor([0, 2, 3, 5]),
        offset=torch.arange(5),
        valid=None,
        size=(2, 2),
    )
    other = VarLenTensor(
        data=torch.arange(6),
        offset=torch.arange(7),
        valid=None,
        size=(2, 3),
    )

    assert not tensor.equal(other)
    assert not torch.equal(tensor, other)
    assert not tensor.allclose(other)
    assert not torch.allclose(tensor, other)

    assert tensor.equal(other[:, ::2])
    assert torch.equal(tensor, other[:, ::2])
    assert tensor.allclose(other[:, ::2])
    assert torch.allclose(tensor, other[:, ::2])


def test_isnan_isfinite() -> None:
    tensor = VarLenTensor.from_list([[1, 2], None, [3]])
    assert tensor.is_nullable

    out = torch.isnan(tensor)
    assert out.dtype == torch.bool
    assert out.equal(torch.tensor([False, True, False]))

    out = tensor.isfinite()
    assert out.dtype == torch.bool
    assert out.equal(torch.tensor([True, False, True]))


def test_masked_select() -> None:
    tensor = VarLenTensor(
        data=torch.arange(9),
        offset=torch.tensor([0, 1, 3, 4, 6, 7, 9]),
        valid=None,
        size=(2, 3),
    )
    mask = torch.tensor([[True, False, True], [False, True, True]])

    out = tensor.masked_select(mask)
    assert isinstance(out, VarLenTensor)
    assert out.size() == (4,)
    assert out.stride() == (1,)
    assert out.storage_offset() == 0
    assert out._data.equal(torch.tensor([0, 3, 6, 7, 8]))
    assert out._offset.equal(torch.tensor([0, 1, 2, 3, 5]))

    out = torch.masked_select(tensor, torch.tensor([[True, False, True]]))
    assert isinstance(out, VarLenTensor)
    assert out._data.equal(torch.tensor([0, 3, 4, 5, 7, 8]))

    out = tensor.masked_select(torch.zeros(2, 3, dtype=torch.bool))
    assert isinstance(out, VarLenTensor)
    assert out.size() == (0,)
    assert out.stride() == (1,)
    assert out._data.equal(torch.empty(0))
    assert out._offset.equal(torch.tensor([0]))


def test_indexing() -> None:
    tensor = VarLenTensor(
        data=torch.arange(9),
        offset=torch.tensor([0, 1, 3, 4, 6, 7, 9]),
        valid=None,
        size=(2, 3),
    )

    out = tensor.index_select(dim=1, index=torch.tensor([2, 0]))
    assert isinstance(out, VarLenTensor)
    assert out.size() == (2, 2)
    assert out.stride() == (2, 1)
    assert out.storage_offset() == 0
    assert out._data.equal(torch.tensor([3, 0, 7, 8, 4, 5]))
    assert out._offset.equal(torch.tensor([0, 1, 2, 4, 6]))

    out = torch.index_select(tensor, dim=0, index=torch.tensor([1, 1, 0]))
    assert isinstance(out, VarLenTensor)
    assert out.size() == (3, 3)
    assert out.stride() == (3, 1)
    assert out._data.equal(
        torch.tensor([4, 5, 6, 7, 8, 4, 5, 6, 7, 8, 0, 1, 2, 3])
    )

    out = tensor.take(torch.tensor([[0, 3], [5, 1]]))
    assert isinstance(out, VarLenTensor)
    assert out.size() == (2, 2)
    assert out.stride() == (2, 1)
    assert out._data.equal(torch.tensor([0, 4, 5, 7, 8, 1, 2]))

    mask = torch.tensor([[True, False, True], [False, True, False]])
    out = tensor[mask]
    assert isinstance(out, VarLenTensor)
    assert out.size() == (3,)
    assert out.stride() == (1,)
    assert out._data.equal(torch.tensor([0, 3, 6]))

    out = tensor[:, torch.tensor([True, False, True])]
    assert isinstance(out, VarLenTensor)
    assert out.size() == (2, 2)
    assert out.stride() == (2, 1)
    assert out._data.equal(torch.tensor([0, 3, 4, 5, 7, 8]))


def test_cat() -> None:
    tensors: list[Tensor] = [
        VarLenTensor(
            data=torch.arange(4),
            offset=torch.arange(5),
            valid=None,
            size=(2, 2),
        ),
        VarLenTensor(
            data=torch.tensor([4, 5, 6]),
            offset=torch.tensor([0, 1, 3]),
            valid=None,
            size=(1, 2),
        ),
    ]

    out = torch.cat(tensors)
    assert isinstance(out, VarLenTensor)
    assert out.size() == (3, 2)
    assert out.stride() == (2, 1)
    assert out.storage_offset() == 0
    assert out._data.equal(torch.arange(7))
    assert out._offset.equal(torch.tensor([0, 1, 2, 3, 4, 5, 7]))

    tensors = [
        VarLenTensor(
            data=torch.arange(4),
            offset=torch.arange(5),
            valid=None,
            size=(2, 2),
        ),
        VarLenTensor(
            data=torch.tensor([4, 5, 6]),
            offset=torch.tensor([0, 1, 3]),
            valid=None,
            size=(2, 1),
        ),
    ]
    out = torch.cat(tensors, dim=-1)
    assert isinstance(out, VarLenTensor)
    assert out.size() == (2, 3)
    assert out.stride() == (3, 1)
    assert out.storage_offset() == 0
    assert out._data.equal(torch.tensor([0, 1, 4, 2, 3, 5, 6]))
    assert out._offset.equal(torch.tensor([0, 1, 2, 3, 4, 5, 7]))


def test_stack() -> None:
    tensors: list[Tensor] = [
        VarLenTensor(
            data=torch.arange(3),
            offset=torch.tensor([0, 1, 3]),
            valid=None,
            size=(2,),
        ),
        VarLenTensor(
            data=torch.arange(3, 5),
            offset=torch.tensor([0, 1, 2]),
            valid=None,
            size=(2,),
        ),
    ]

    out = torch.stack(tensors)
    assert isinstance(out, VarLenTensor)
    assert out.size() == (2, 2)
    assert out.stride() == (2, 1)
    assert out.storage_offset() == 0
    assert out._data.equal(torch.arange(5))
    assert out._offset.equal(torch.tensor([0, 1, 3, 4, 5]))

    out = torch.stack(tensors, dim=-1)
    assert isinstance(out, VarLenTensor)
    assert out.size() == (2, 2)
    assert out.stride() == (2, 1)
    assert out.storage_offset() == 0
    assert out._data.equal(torch.tensor([0, 3, 1, 2, 4]))
    assert out._offset.equal(torch.tensor([0, 1, 2, 4, 5]))


def test_pin_memory() -> None:
    tensor = VarLenTensor(
        data=torch.arange(4),
        offset=torch.arange(5),
        valid=None,
        size=(4,),
    )

    assert not tensor.is_pinned()


@onlyCUDA
def test_pin_memory_cuda() -> None:
    tensor = VarLenTensor(
        data=torch.arange(4),
        offset=torch.arange(5),
        valid=None,
        size=(4,),
    )

    assert tensor.pin_memory().is_pinned()


def test_share_memory() -> None:
    tensor = VarLenTensor(
        data=torch.arange(4),
        offset=torch.arange(5),
        valid=None,
        size=(4,),
    )

    assert not tensor.is_shared()
    try:
        tensor.share_memory_()
        assert tensor.is_shared()
    except RuntimeError:
        pass


def test_view() -> None:
    tensor = VarLenTensor(
        data=torch.arange(8),
        offset=torch.arange(9),
        valid=None,
        size=(2, 4),
    )

    out = tensor.view(4, 2)
    assert isinstance(out, VarLenTensor)
    assert out.size() == (4, 2)
    assert out.stride() == (2, 1)
    assert out.storage_offset() == 0
    assert out._data.data_ptr() == tensor._data.data_ptr()
    assert out._offset.data_ptr() == tensor._offset.data_ptr()
    assert torch._C._is_alias_of(  # ty: ignore[unresolved-attribute]
        out,
        tensor,
    )

    out = tensor.view(1, -1)
    assert isinstance(out, VarLenTensor)
    assert out.size() == (1, 8)
    assert out.stride() == (8, 1)
    assert out.storage_offset() == 0

    tensor = VarLenTensor(
        data=torch.arange(8),
        offset=torch.arange(9),
        valid=None,
        size=(2, 2),
        stride=(1, 4),
    )
    with pytest.raises(RuntimeError, match="view size is not compatible"):
        tensor.view(4)


def test_squeeze() -> None:
    tensor = VarLenTensor(
        data=torch.arange(8),
        offset=torch.arange(9),
        valid=None,
        size=(1, 2, 1, 4),
        storage_offset=0,
    )

    out = tensor.squeeze()
    assert isinstance(out, VarLenTensor)
    assert out.size() == (2, 4)
    assert out.stride() == (4, 1)
    assert out.storage_offset() == 0
    assert out._data.data_ptr() == tensor._data.data_ptr()
    assert out._offset.data_ptr() == tensor._offset.data_ptr()

    out = tensor.squeeze(0)
    assert isinstance(out, VarLenTensor)
    assert out.size() == (2, 1, 4)
    assert out.stride() == (4, 4, 1)
    assert out.storage_offset() == 0

    out = tensor.squeeze((0, 2))
    assert isinstance(out, VarLenTensor)
    assert out.size() == (2, 4)
    assert out.stride() == (4, 1)
    assert out.storage_offset() == 0

    out = tensor.squeeze(1)
    assert isinstance(out, VarLenTensor)
    assert out.size() == (1, 2, 1, 4)
    assert out.stride() == (8, 4, 4, 1)
    assert out.storage_offset() == 0


def test_unsqueeze() -> None:
    tensor = VarLenTensor(
        data=torch.arange(8),
        offset=torch.arange(9),
        valid=None,
        size=(2, 2),
        stride=(1, 4),
        storage_offset=1,
    )

    for dim in range(-tensor.dim() - 1, tensor.dim() + 1):
        out = tensor.unsqueeze(dim)
        assert isinstance(out, VarLenTensor)
        assert out.storage_offset() == 1
        assert out._data.data_ptr() == tensor._data.data_ptr()
        assert out._offset.data_ptr() == tensor._offset.data_ptr()

    out = tensor.unsqueeze(0)
    assert out.size() == (1, 2, 2)
    assert out.stride() == (2, 1, 4)

    out = tensor.unsqueeze(1)
    assert out.size() == (2, 1, 2)
    assert out.stride() == (1, 8, 4)

    out = tensor.unsqueeze(2)
    assert out.size() == (2, 2, 1)
    assert out.stride() == (1, 4, 1)

    scalar = VarLenTensor(
        data=torch.arange(8),
        offset=torch.arange(9),
        valid=None,
        size=(),
    )
    out = scalar.unsqueeze(0)
    assert isinstance(out, VarLenTensor)
    assert out.size() == (1,)
    assert out.stride() == (1,)

    with pytest.raises(IndexError, match="Dimension out of range"):
        tensor.unsqueeze(tensor.dim() + 1)


def test_transpose_permute() -> None:
    tensor = VarLenTensor(
        data=torch.arange(12),
        offset=torch.arange(13),
        valid=None,
        size=(2, 3),
        stride=(1, 4),
        storage_offset=2,
    )

    out = tensor.t()
    assert isinstance(out, VarLenTensor)
    assert out.size() == (3, 2)
    assert out.stride() == (4, 1)
    assert out.storage_offset() == 2
    assert out._data.data_ptr() == tensor._data.data_ptr()
    assert out._offset.data_ptr() == tensor._offset.data_ptr()

    out = tensor.transpose(0, 1)
    assert isinstance(out, VarLenTensor)
    assert out.size() == (3, 2)
    assert out.stride() == (4, 1)
    assert out.storage_offset() == 2
    assert out._data.data_ptr() == tensor._data.data_ptr()
    assert out._offset.data_ptr() == tensor._offset.data_ptr()

    tensor = VarLenTensor(
        data=torch.arange(24),
        offset=torch.arange(25),
        valid=None,
        size=(2, 3, 4),
        stride=(12, 4, 1),
    )

    out = tensor.transpose(0, -1)
    assert isinstance(out, VarLenTensor)
    assert out.size() == (4, 3, 2)
    assert out.stride() == (1, 4, 12)
    assert out.storage_offset() == 0
    assert out._data.data_ptr() == tensor._data.data_ptr()
    assert out._offset.data_ptr() == tensor._offset.data_ptr()

    out = tensor.permute(2, 0, 1)
    assert isinstance(out, VarLenTensor)
    assert out.size() == (4, 2, 3)
    assert out.stride() == (1, 12, 4)
    assert out.storage_offset() == 0
    assert out._data.data_ptr() == tensor._data.data_ptr()
    assert out._offset.data_ptr() == tensor._offset.data_ptr()

    with pytest.raises(RuntimeError, match="t\\(\\) expects"):
        tensor.t()


def test_select_slice_narrow_expand() -> None:
    tensor = VarLenTensor(
        data=torch.arange(24),
        offset=torch.arange(25),
        valid=None,
        size=(2, 3, 4),
    )

    out = tensor.select(dim=1, index=1)
    assert isinstance(out, VarLenTensor)
    assert out.size() == (2, 4)
    assert out.stride() == (12, 1)
    assert out.storage_offset() == 4
    assert out._data.data_ptr() == tensor._data.data_ptr()
    assert out._offset.data_ptr() == tensor._offset.data_ptr()

    out = tensor[:, 1:3, ::2]
    assert isinstance(out, VarLenTensor)
    assert out.size() == (2, 2, 2)
    assert out.stride() == (12, 4, 2)
    assert out.storage_offset() == 4
    assert out._data.data_ptr() == tensor._data.data_ptr()
    assert out._offset.data_ptr() == tensor._offset.data_ptr()

    out = tensor.narrow(dim=1, start=1, length=2)
    assert isinstance(out, VarLenTensor)
    assert out.size() == (2, 2, 4)
    assert out.stride() == (12, 4, 1)
    assert out.storage_offset() == 4
    assert out._data.data_ptr() == tensor._data.data_ptr()
    assert out._offset.data_ptr() == tensor._offset.data_ptr()

    tensor = VarLenTensor(
        data=torch.arange(4),
        offset=torch.arange(5),
        valid=None,
        size=(1, 4),
    )
    out = tensor.expand(3, 4)
    assert isinstance(out, VarLenTensor)
    assert out.size() == (3, 4)
    assert out.stride() == (0, 1)
    assert out.storage_offset() == 0
    assert out._data.data_ptr() == tensor._data.data_ptr()
    assert out._offset.data_ptr() == tensor._offset.data_ptr()
    out = out.contiguous()
    assert out.is_contiguous()
    assert isinstance(out, VarLenTensor)
    assert out.size() == (3, 4)
    assert out.stride() == (4, 1)
    assert out.storage_offset() == 0
    assert out._data.equal(torch.tensor([0, 1, 2, 3] * 3))
    assert out._offset.equal(torch.arange(13))


def test_unbind() -> None:
    tensor = VarLenTensor(
        data=torch.arange(25),
        offset=torch.arange(26),
        valid=None,
        size=(2, 3, 4),
        storage_offset=1,
    )

    out = tensor.unbind(dim=1)
    assert isinstance(out, tuple)
    assert len(out) == 3
    assert out[1].size() == (2, 4)
    assert out[1].stride() == (12, 1)
    assert out[1].storage_offset() == 5
    assert isinstance(out[1], VarLenTensor)
    assert out[1]._data.data_ptr() == tensor._data.data_ptr()
    assert out[1]._offset.data_ptr() == tensor._offset.data_ptr()
    assert all(
        torch._C._is_alias_of(tensor, value)  # ty: ignore[unresolved-attribute]
        for value in out
    )

    out = tuple(tensor)
    assert len(out) == 2
    assert out[1].size() == (3, 4)
    assert out[1].stride() == (4, 1)
    assert out[1].storage_offset() == 13


def test_split() -> None:
    tensor = VarLenTensor(
        data=torch.arange(25),
        offset=torch.arange(26),
        valid=None,
        size=(2, 3, 4),
        storage_offset=1,
    )

    out = tensor.split(2, dim=2)
    assert len(out) == 2
    assert out[0].size() == (2, 3, 2)
    assert out[0].stride() == (12, 4, 1)
    assert out[0].storage_offset() == 1
    assert out[1].size() == (2, 3, 2)
    assert out[1].stride() == (12, 4, 1)
    assert out[1].storage_offset() == 3
    assert isinstance(out[1], VarLenTensor)
    assert out[1]._data.data_ptr() == tensor._data.data_ptr()
    assert out[1]._offset.data_ptr() == tensor._offset.data_ptr()
    assert all(
        torch._C._is_alias_of(tensor, value)  # ty: ignore[unresolved-attribute]
        for value in out
    )

    out = tensor.split([1, 2], dim=-2)
    assert len(out) == 2
    assert out[0].size() == (2, 1, 4)
    assert out[0].storage_offset() == 1
    assert out[1].size() == (2, 2, 4)
    assert out[1].storage_offset() == 5
    assert all(
        torch._C._is_alias_of(tensor, value)  # ty: ignore[unresolved-attribute]
        for value in out
    )


def test_unsafe_view() -> None:
    tensor = VarLenTensor(
        data=torch.arange(12),
        offset=torch.arange(13),
        valid=None,
        size=(3, 2),
        stride=(4, 1),
    )

    out = tensor.reshape(2, 3)
    assert isinstance(out, VarLenTensor)
    assert out.size() == (2, 3)
    assert out.stride() == (3, 1)
    assert out.storage_offset() == 0
    assert out._data.equal(torch.tensor([0, 1, 4, 5, 8, 9]))
    assert out._offset.equal(torch.arange(7))

    out = tensor.flatten()
    assert isinstance(out, VarLenTensor)
    assert out.size() == (6,)
    assert out.stride() == (1,)
    assert out.storage_offset() == 0
    assert out._data.equal(torch.tensor([0, 1, 4, 5, 8, 9]))
    assert out._offset.equal(torch.arange(7))


def test_reshape_inference_mode() -> None:
    tensor = VarLenTensor(
        data=torch.arange(12),
        offset=torch.arange(13),
        valid=None,
        size=(3, 2),
        stride=(4, 1),
    )

    with torch.inference_mode():
        view = tensor.reshape(3, 2)
        copied = tensor.reshape(2, 3)

    assert not torch.is_inference(view)
    assert torch.is_inference(copied)
