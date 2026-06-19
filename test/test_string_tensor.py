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

    tensor = StringTensor.from_strings(
        [["hi", "é"], ["", "abc"]],
        offset_dtype=torch.int32,
    )
    assert tensor._offset.dtype == torch.int32
    assert tensor.to_arrow().type == pa.string()


def test_arrow() -> None:
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

    tensor = StringTensor.from_strings([["hi", "é"], ["", "abc"]])
    array = tensor.to_arrow()
    assert array.type == pa.large_string()
    assert array.to_pylist() == ["hi", "é", "", "abc"]


def test_pandas() -> None:
    pd = pytest.importorskip("pandas")

    tensor = StringTensor.from_pandas(pd.Series(["hi", "é", "", None]))
    assert tensor.size() == (4,)
    assert tensor.stride() == (1,)
    assert tensor.dtype == torch.uint8
    assert tensor._data.equal(torch.tensor([104, 105, 195, 169]))
    assert tensor._offset.equal(torch.tensor([0, 2, 4, 4, 4]))

    tensor = StringTensor.from_pandas(
        pd.Series(["hi", "é", "", None]),
        offset_dtype=torch.int32,
    )
    assert tensor._offset.dtype == torch.int32

    series = tensor.to_pandas()
    assert isinstance(series, pd.Series)
    assert series.tolist() == ["hi", "é", "", ""]


def test_offset_dtype() -> None:
    tensor = StringTensor.from_arrow(pa.array(["hi", "é"], type=pa.string()))
    assert tensor._offset.dtype == torch.int32
    assert tensor.to_arrow().type == pa.string()

    tensor = StringTensor.from_arrow(
        pa.array(["hi", "é"], type=pa.large_string())
    )
    assert tensor._offset.dtype == torch.int64
    assert tensor.to_arrow().type == pa.large_string()


def test_allowed_dtype() -> None:
    tensor = StringTensor.from_strings("hi")

    with pytest.raises(TypeError, match="Cannot convert"):
        tensor.to(torch.float32)


def test_item() -> None:
    assert StringTensor.from_strings("é").item() == "é"
    assert StringTensor.from_strings(["hi", "é"])[1].item() == "é"
    assert str(StringTensor.from_strings([""])) == ""

    with pytest.raises(RuntimeError, match="cannot be converted"):
        StringTensor.from_strings(["hi", "é"]).item()


def test_tolist() -> None:
    for strings in [
        "é",
        ["hi", "é"],
        [],
        [["a", "bb", "c"], ["dd", "e", "ff"]],
    ]:
        assert StringTensor.from_strings(strings).tolist() == strings
