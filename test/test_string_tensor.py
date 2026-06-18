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
    tensor = StringTensor.from_strings([["hi", "é"], ["", "abc"]])

    out = tensor.clone()
    assert isinstance(out, StringTensor)
    assert out.size() == tensor.size()
    assert out.stride() == tensor.stride()
    assert out._data.equal(tensor._data)
    assert out._offset.equal(tensor._offset)
    assert out._data.data_ptr() != tensor._data.data_ptr()
    assert out._offset.data_ptr() != tensor._offset.data_ptr()

    non_contiguous = StringTensor(
        data=tensor._data,
        offset=tensor._offset,
        size=tensor.size(),
        stride=(1, 2),
    )
    with pytest.raises(NotImplementedError, match="non-contiguous"):
        non_contiguous.clone()
    with pytest.raises(NotImplementedError, match="non-contiguous"):
        non_contiguous.to("cpu", copy=True)

    tensor = StringTensor.from_arrow(
        pa.array(["x", "hi", "abc"], type=pa.large_string()).slice(1)
    )
    assert tensor.storage_offset() == 1
    out = tensor.clone()
    assert isinstance(out, StringTensor)
    assert out.storage_offset() == 0
    assert out._data.equal(torch.tensor([104, 105, 97, 98, 99]))
    assert out._offset.equal(torch.tensor([0, 2, 5]))

    out = tensor.to("cpu", copy=True)
    assert isinstance(out, StringTensor)
    assert out.storage_offset() == 0
    assert out._data.equal(torch.tensor([104, 105, 97, 98, 99]))
    assert out._offset.equal(torch.tensor([0, 2, 5]))
    assert out._data.data_ptr() != tensor._data.data_ptr()
    assert out._offset.data_ptr() != tensor._offset.data_ptr()

    with pytest.raises(TypeError, match="Cannot convert"):
        tensor.to(torch.float32)

    out = torch.ops.aten._to_copy.default(tensor, pin_memory=True)
    assert isinstance(out, StringTensor)
    assert not out.is_pinned()

    assert not tensor.is_pinned()
    assert not torch.ops.aten.is_pinned.default(tensor, None)
    if torch.cuda.is_available():
        assert tensor.pin_memory().is_pinned()

    assert not tensor.is_shared()
    try:
        out.share_memory_()
        assert out.is_shared()
    except RuntimeError:
        pass
