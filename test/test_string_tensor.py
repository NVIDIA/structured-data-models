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

    base = StringTensor.from_strings(
        [
            ["a", "b", "c", "d"],
            ["e", "f", "g", "h"],
            ["i", "j", "k", "l"],
            ["m", "n", "o", "p"],
        ]
    )

    dense = StringTensor(
        data=base._data,
        offset=base._offset,
        size=(4, 2),
        stride=(1, 4),
        storage_offset=2,
    )
    out = dense.clone()
    assert isinstance(out, StringTensor)
    assert out.stride() == dense.stride()
    assert out._data.equal(torch.tensor(list(b"cdefghij"), dtype=torch.uint8))
    assert out._offset.equal(torch.arange(9))

    out = dense.clone(memory_format=torch.contiguous_format)
    assert isinstance(out, StringTensor)
    assert out.stride() == (2, 1)
    assert out._data.equal(torch.tensor(list(b"cgdheifj"), dtype=torch.uint8))
    assert out._offset.equal(torch.arange(9))

    gapped = StringTensor(
        data=base._data,
        offset=base._offset,
        size=(4, 2),
        stride=(4, 1),
        storage_offset=2,
    )
    out = gapped.clone()
    assert isinstance(out, StringTensor)
    assert out.stride() == (2, 1)
    assert out._data.equal(torch.tensor(list(b"cdghklop"), dtype=torch.uint8))
    assert out._offset.equal(torch.arange(9))

    overlapping = StringTensor(
        data=base._data,
        offset=base._offset,
        size=(4, 4),
        stride=(0, 1),
    )
    out = overlapping.clone()
    assert isinstance(out, StringTensor)
    assert out.stride() == (4, 1)
    assert out._data.equal(
        torch.tensor(list(b"abcdabcdabcdabcd"), dtype=torch.uint8)
    )
    assert out._offset.equal(torch.arange(17))

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

    assert not tensor.is_pinned()
    if torch.cuda.is_available():
        assert tensor.pin_memory().is_pinned()

    assert not tensor.is_shared()
    try:
        out.share_memory_()
        assert out.is_shared()
    except RuntimeError:
        pass
