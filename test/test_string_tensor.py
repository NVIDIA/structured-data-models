import pyarrow as pa
import pytest
import torch
from schemafm import StringTensor


def test_from_strings() -> None:
    tensor = StringTensor.from_strings([["hi", "é"], ["", "abc"]])
    assert tensor.size() == (2, 2)
    assert tensor.stride() == (2, 1)
    assert tensor.dtype == torch.uint8
    assert tensor._data.equal(torch.tensor([104, 105, 195, 169, 97, 98, 99]))
    assert tensor._offset.equal(torch.tensor([0, 2, 4, 4, 7]))

    tensor = StringTensor.from_strings([])
    assert tensor.size() == (0,)
    assert tensor.stride() == (1,)
    assert tensor.dtype == torch.uint8
    assert tensor._data.equal(torch.tensor([]))
    assert tensor._offset.equal(torch.tensor([0]))

    tensor = StringTensor.from_strings("hi")
    assert tensor.size() == ()
    assert tensor.stride() == ()
    assert tensor.dtype == torch.uint8
    assert tensor._data.equal(torch.tensor([104, 105]))
    assert tensor._offset.equal(torch.tensor([0, 2]))


def test_from_arrow() -> None:
    tensor = StringTensor.from_arrow(pa.array(["hi", "é", "", None]))
    assert tensor.size() == (4,)
    assert tensor.stride() == (1,)
    assert tensor.dtype == torch.uint8
    assert tensor._data.equal(torch.tensor([104, 105, 195, 169]))
    assert tensor._offset.equal(torch.tensor([0, 2, 4, 4, 4]))

    tensor = StringTensor.from_arrow(pa.array([], type=pa.string()))
    assert tensor.size() == (0,)
    assert tensor.stride() == (1,)
    assert tensor.dtype == torch.uint8
    assert tensor._data.equal(torch.tensor([]))
    assert tensor._offset.equal(torch.tensor([0]))


def test_from_pandas() -> None:
    pd = pytest.importorskip("pandas")

    tensor = StringTensor.from_pandas(pd.Series(["hi", "é", "", None]))
    assert tensor.size() == (4,)
    assert tensor.stride() == (1,)
    assert tensor.dtype == torch.uint8
    assert tensor._data.equal(torch.tensor([104, 105, 195, 169]))
    assert tensor._offset.equal(torch.tensor([0, 2, 4, 4, 4]))


def test_to_copy() -> None:
    data = torch.arange(16, dtype=torch.uint8)
    offset = torch.arange(data.numel() + 1)

    tensor = StringTensor(
        data=data,
        offset=offset,
        size=(4, 4),
    )
    out = tensor.clone()
    assert isinstance(out, StringTensor)
    assert out.size() == tensor.size()
    assert out.stride() == tensor.stride()
    assert out._data.equal(tensor._data)
    assert out._offset.equal(tensor._offset)
    assert out._data.data_ptr() != tensor._data.data_ptr()
    assert out._offset.data_ptr() != tensor._offset.data_ptr()

    tensor = StringTensor(
        data=data,
        offset=offset,
        size=(4, 2),
        stride=(1, 4),
        storage_offset=2,
    )
    out = tensor.clone()
    assert isinstance(out, StringTensor)
    assert out.stride() == tensor.stride()
    assert out._data.equal(torch.arange(2, 10, dtype=torch.uint8))
    assert out._offset.equal(torch.arange(9))
    assert out._data.data_ptr() != tensor._data.data_ptr()
    assert out._offset.data_ptr() != tensor._offset.data_ptr()

    out = tensor.clone(memory_format=torch.contiguous_format)
    assert isinstance(out, StringTensor)
    assert out.stride() == (2, 1)
    assert out._data.equal(torch.tensor([2, 6, 3, 7, 4, 8, 5, 9]))
    assert out._offset.equal(torch.arange(9))
    assert out._data.data_ptr() != tensor._data.data_ptr()
    assert out._offset.data_ptr() != tensor._offset.data_ptr()

    tensor = StringTensor(
        data=data,
        offset=offset,
        size=(4, 2),
        stride=(4, 1),
        storage_offset=2,
    )
    out = tensor.clone()
    assert isinstance(out, StringTensor)
    assert out.stride() == (2, 1)
    assert out._data.equal(torch.tensor([2, 3, 6, 7, 10, 11, 14, 15]))
    assert out._offset.equal(torch.arange(9))
    assert out._data.data_ptr() != tensor._data.data_ptr()
    assert out._offset.data_ptr() != tensor._offset.data_ptr()

    tensor = StringTensor(
        data=data,
        offset=offset,
        size=(4, 4),
        stride=(0, 1),
    )
    out = tensor.clone()
    assert isinstance(out, StringTensor)
    assert out.stride() == (4, 1)
    assert out._data.equal(torch.tensor([0, 1, 2, 3] * 4, dtype=torch.uint8))
    assert out._offset.equal(torch.arange(17))
    assert out._data.data_ptr() != tensor._data.data_ptr()
    assert out._offset.data_ptr() != tensor._offset.data_ptr()

    tensor = StringTensor(
        data=data,
        offset=offset,
        size=(2,),
        storage_offset=2,
    )
    out = tensor.clone()
    assert isinstance(out, StringTensor)
    assert out.storage_offset() == 0
    assert out._data.equal(torch.tensor([2, 3], dtype=torch.uint8))
    assert out._offset.equal(torch.tensor([0, 1, 2]))
    assert out._data.data_ptr() != tensor._data.data_ptr()
    assert out._offset.data_ptr() != tensor._offset.data_ptr()

    with pytest.raises(TypeError, match="Cannot convert"):
        tensor.to(torch.float32)


def test_pin_memory() -> None:
    tensor = StringTensor.from_strings(["hi", "abc"])

    assert not tensor.is_pinned()
    if torch.cuda.is_available():
        assert tensor.pin_memory().is_pinned()


def test_share_memory() -> None:
    tensor = StringTensor.from_strings(["hi", "abc"])

    assert not tensor.is_shared()
    try:
        tensor.share_memory_()
        assert tensor.is_shared()
    except RuntimeError:
        pass


def test_view() -> None:
    tensor = StringTensor(
        data=torch.arange(8, dtype=torch.uint8),
        offset=torch.arange(9),
        size=(2, 4),
    )

    out = tensor.view(4, 2)
    assert isinstance(out, StringTensor)
    assert out.size() == (4, 2)
    assert out.stride() == (2, 1)
    assert out.storage_offset() == 0
    assert out._data.data_ptr() == tensor._data.data_ptr()
    assert out._offset.data_ptr() == tensor._offset.data_ptr()

    out = tensor.view(1, -1)
    assert isinstance(out, StringTensor)
    assert out.size() == (1, 8)
    assert out.stride() == (8, 1)
    assert out.storage_offset() == 0

    tensor = StringTensor(
        data=torch.arange(8, dtype=torch.uint8),
        offset=torch.arange(9),
        size=(2, 2),
        stride=(1, 4),
    )
    with pytest.raises(RuntimeError, match="view size is not compatible"):
        tensor.view(4)


def test_squeeze() -> None:
    tensor = StringTensor(
        data=torch.arange(8, dtype=torch.uint8),
        offset=torch.arange(9),
        size=(1, 2, 1, 4),
        storage_offset=0,
    )

    out = tensor.squeeze()
    assert isinstance(out, StringTensor)
    assert out.size() == (2, 4)
    assert out.stride() == (4, 1)
    assert out.storage_offset() == 0
    assert out._data.data_ptr() == tensor._data.data_ptr()
    assert out._offset.data_ptr() == tensor._offset.data_ptr()

    out = tensor.squeeze(0)
    assert isinstance(out, StringTensor)
    assert out.size() == (2, 1, 4)
    assert out.stride() == (4, 4, 1)
    assert out.storage_offset() == 0

    out = tensor.squeeze((0, 2))
    assert isinstance(out, StringTensor)
    assert out.size() == (2, 4)
    assert out.stride() == (4, 1)
    assert out.storage_offset() == 0

    out = tensor.squeeze(1)
    assert isinstance(out, StringTensor)
    assert out.size() == (1, 2, 1, 4)
    assert out.stride() == (8, 4, 4, 1)
    assert out.storage_offset() == 0


def test_unsqueeze() -> None:
    tensor = StringTensor(
        data=torch.arange(8, dtype=torch.uint8),
        offset=torch.arange(9),
        size=(2, 2),
        stride=(1, 4),
        storage_offset=1,
    )

    for dim in range(-tensor.dim() - 1, tensor.dim() + 1):
        out = tensor.unsqueeze(dim)
        assert isinstance(out, StringTensor)
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

    scalar = StringTensor(
        data=torch.arange(8, dtype=torch.uint8),
        offset=torch.arange(9),
        size=(),
    )
    out = scalar.unsqueeze(0)
    assert isinstance(out, StringTensor)
    assert out.size() == (1,)
    assert out.stride() == (1,)

    with pytest.raises(IndexError, match="Dimension out of range"):
        tensor.unsqueeze(tensor.dim() + 1)


def test_transpose_permute() -> None:
    tensor = StringTensor(
        data=torch.arange(12, dtype=torch.uint8),
        offset=torch.arange(13),
        size=(2, 3),
        stride=(1, 4),
        storage_offset=2,
    )

    out = tensor.t()
    assert isinstance(out, StringTensor)
    assert out.size() == (3, 2)
    assert out.stride() == (4, 1)
    assert out.storage_offset() == 2
    assert out._data.data_ptr() == tensor._data.data_ptr()
    assert out._offset.data_ptr() == tensor._offset.data_ptr()

    out = tensor.transpose(0, 1)
    assert isinstance(out, StringTensor)
    assert out.size() == (3, 2)
    assert out.stride() == (4, 1)
    assert out.storage_offset() == 2
    assert out._data.data_ptr() == tensor._data.data_ptr()
    assert out._offset.data_ptr() == tensor._offset.data_ptr()

    tensor = StringTensor(
        data=torch.arange(24, dtype=torch.uint8),
        offset=torch.arange(25),
        size=(2, 3, 4),
        stride=(12, 4, 1),
    )

    out = tensor.transpose(0, -1)
    assert isinstance(out, StringTensor)
    assert out.size() == (4, 3, 2)
    assert out.stride() == (1, 4, 12)
    assert out.storage_offset() == 0
    assert out._data.data_ptr() == tensor._data.data_ptr()
    assert out._offset.data_ptr() == tensor._offset.data_ptr()

    out = tensor.permute(2, 0, 1)
    assert isinstance(out, StringTensor)
    assert out.size() == (4, 2, 3)
    assert out.stride() == (1, 12, 4)
    assert out.storage_offset() == 0
    assert out._data.data_ptr() == tensor._data.data_ptr()
    assert out._offset.data_ptr() == tensor._offset.data_ptr()

    with pytest.raises(RuntimeError, match="t\\(\\) expects"):
        tensor.t()


def test_select_slice_narrow_expand() -> None:
    tensor = StringTensor(
        data=torch.arange(24, dtype=torch.uint8),
        offset=torch.arange(25),
        size=(2, 3, 4),
    )

    out = tensor.select(dim=1, index=1)
    assert isinstance(out, StringTensor)
    assert out.size() == (2, 4)
    assert out.stride() == (12, 1)
    assert out.storage_offset() == 4
    assert out._data.data_ptr() == tensor._data.data_ptr()
    assert out._offset.data_ptr() == tensor._offset.data_ptr()

    out = tensor[:, 1:3, ::2]
    assert isinstance(out, StringTensor)
    assert out.size() == (2, 2, 2)
    assert out.stride() == (12, 4, 2)
    assert out.storage_offset() == 4
    assert out._data.data_ptr() == tensor._data.data_ptr()
    assert out._offset.data_ptr() == tensor._offset.data_ptr()

    out = tensor.narrow(dim=1, start=1, length=2)
    assert isinstance(out, StringTensor)
    assert out.size() == (2, 2, 4)
    assert out.stride() == (12, 4, 1)
    assert out.storage_offset() == 4
    assert out._data.data_ptr() == tensor._data.data_ptr()
    assert out._offset.data_ptr() == tensor._offset.data_ptr()

    tensor = StringTensor(
        data=torch.arange(4, dtype=torch.uint8),
        offset=torch.arange(5),
        size=(1, 4),
    )
    out = tensor.expand(3, 4)
    assert isinstance(out, StringTensor)
    assert out.size() == (3, 4)
    assert out.stride() == (0, 1)
    assert out.storage_offset() == 0
    assert out._data.data_ptr() == tensor._data.data_ptr()
    assert out._offset.data_ptr() == tensor._offset.data_ptr()
    out = out.clone()
