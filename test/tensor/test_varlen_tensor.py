from typing import cast

import pytest
import torch
from schemafm import VarLenTensor
from torch import Tensor


def test_dtype_conversion() -> None:
    tensor = VarLenTensor.from_tensor(torch.arange(4).view(2, 2))

    out = tensor.to(torch.float64)
    assert isinstance(out, VarLenTensor)
    assert out.dtype == torch.float64
    assert out._data.dtype == torch.float64
    assert out._offset.dtype == torch.int64


def test_autograd() -> None:
    data = torch.arange(4, dtype=torch.float32, requires_grad=True)
    tensor = VarLenTensor(
        data=data,
        offset=torch.arange(5),
        size=(2, 2),
    )
    assert tensor.requires_grad

    out = tensor.clone()
    assert isinstance(out, VarLenTensor)

    out._data.sum().backward()
    assert data.grad is not None
    assert data.grad.equal(torch.ones_like(data))

    tensor.detach_()
    assert not tensor.requires_grad
    tensor.requires_grad_(True)
    assert tensor.requires_grad
    tensor = tensor.detach()
    assert not tensor.requires_grad


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
                size=(1, 2),
            ),
            VarLenTensor(
                data=torch.tensor([3, 4]),
                offset=torch.tensor([0, 1, 2], dtype=torch.int32),
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
                size=(1, 2),
            ),
            VarLenTensor(
                data=torch.tensor([3, 4]),
                offset=torch.tensor([0, 1, 2], dtype=torch.int64),
                size=(1, 2),
            ),
        ]
    )
    assert isinstance(out, VarLenTensor)
    assert out._offset.dtype == torch.int64


def test_to_copy() -> None:
    data = torch.arange(16)
    offset = torch.arange(data.numel() + 1)

    tensor = VarLenTensor(
        data=data,
        offset=offset,
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

    tensor = VarLenTensor(
        data=data,
        offset=offset,
        size=(4, 2),
        stride=(1, 4),
        storage_offset=2,
    )
    out = tensor.clone()
    assert isinstance(out, VarLenTensor)
    assert out.stride() == tensor.stride()
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


def test_data_offset() -> None:
    tensor = VarLenTensor(
        data=torch.arange(8),
        offset=torch.arange(9),
        size=(2,),
        storage_offset=2,
    )
    data, offset = tensor.data_offset
    assert data.equal(torch.tensor([2, 3]))
    assert offset.equal(torch.tensor([0, 1, 2]))

    tensor = VarLenTensor(
        data=torch.arange(8),
        offset=torch.arange(9),
        size=(2, 4),
    )
    with pytest.raises(RuntimeError, match="non-contiguous"):
        _ = cast(VarLenTensor, tensor[:, ::2]).data_offset


def test_equal_allclose() -> None:
    tensor = VarLenTensor(
        data=torch.tensor([0, 2, 3, 5]),
        offset=torch.arange(5),
        size=(2, 2),
    )
    other = VarLenTensor(
        data=torch.arange(6),
        offset=torch.arange(7),
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


def test_masked_select() -> None:
    tensor = VarLenTensor(
        data=torch.arange(9),
        offset=torch.tensor([0, 1, 3, 4, 6, 7, 9]),
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
            size=(2, 2),
        ),
        VarLenTensor(
            data=torch.tensor([4, 5, 6]),
            offset=torch.tensor([0, 1, 3]),
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
            size=(2, 2),
        ),
        VarLenTensor(
            data=torch.tensor([4, 5, 6]),
            offset=torch.tensor([0, 1, 3]),
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
            size=(2,),
        ),
        VarLenTensor(
            data=torch.arange(3, 5),
            offset=torch.tensor([0, 1, 2]),
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
        size=(4,),
    )

    assert not tensor.is_pinned()
    if torch.cuda.is_available():
        assert tensor.pin_memory().is_pinned()


def test_share_memory() -> None:
    tensor = VarLenTensor(
        data=torch.arange(4),
        offset=torch.arange(5),
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
        size=(2, 4),
    )

    out = tensor.view(4, 2)
    assert isinstance(out, VarLenTensor)
    assert out.size() == (4, 2)
    assert out.stride() == (2, 1)
    assert out.storage_offset() == 0
    assert out._data.data_ptr() == tensor._data.data_ptr()
    assert out._offset.data_ptr() == tensor._offset.data_ptr()

    out = tensor.view(1, -1)
    assert isinstance(out, VarLenTensor)
    assert out.size() == (1, 8)
    assert out.stride() == (8, 1)
    assert out.storage_offset() == 0

    tensor = VarLenTensor(
        data=torch.arange(8),
        offset=torch.arange(9),
        size=(2, 2),
        stride=(1, 4),
    )
    with pytest.raises(RuntimeError, match="view size is not compatible"):
        tensor.view(4)


def test_squeeze() -> None:
    tensor = VarLenTensor(
        data=torch.arange(8),
        offset=torch.arange(9),
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

    out = tuple(tensor)
    assert len(out) == 2
    assert out[1].size() == (3, 4)
    assert out[1].stride() == (4, 1)
    assert out[1].storage_offset() == 13


def test_split() -> None:
    tensor = VarLenTensor(
        data=torch.arange(25),
        offset=torch.arange(26),
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

    out = tensor.split([1, 2], dim=-2)
    assert len(out) == 2
    assert out[0].size() == (2, 1, 4)
    assert out[0].storage_offset() == 1
    assert out[1].size() == (2, 2, 4)
    assert out[1].storage_offset() == 5


def test_unsafe_view() -> None:
    tensor = VarLenTensor(
        data=torch.arange(12),
        offset=torch.arange(13),
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
