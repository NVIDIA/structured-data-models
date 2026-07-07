from typing import Any, cast

import pyarrow as pa
import pytest
import torch
from sdm import StringTensor
from sdm.testing import withCUDA


def test_from_list() -> None:
    tensor = StringTensor.from_list([["hi", "é"], ["", "abc"]])
    assert repr(tensor) == "StringTensor(..., size=(2, 2))"
    assert tensor.size() == (2, 2)
    assert tensor.stride() == (2, 1)
    assert tensor.dtype == torch.uint8
    assert tensor._data.equal(torch.tensor([104, 105, 195, 169, 97, 98, 99]))
    assert tensor._offset.equal(torch.tensor([0, 2, 4, 4, 7]))

    tensor = StringTensor.from_list([])
    assert tensor.size() == (0,)
    assert tensor.stride() == (1,)
    assert tensor.dtype == torch.uint8
    assert tensor._data.equal(torch.tensor([]))
    assert tensor._offset.equal(torch.tensor([0]))

    tensor = StringTensor.from_list("hi")
    assert tensor.size() == ()
    assert tensor.stride() == ()
    assert tensor.dtype == torch.uint8
    assert tensor._data.equal(torch.tensor([104, 105]))
    assert tensor._offset.equal(torch.tensor([0, 2]))


def test_arrow() -> None:
    tensor = StringTensor.from_arrow(pa.array(["hi", "é", ""]))
    assert tensor.size() == (3,)
    assert tensor.stride() == (1,)
    assert tensor.dtype == torch.uint8
    assert tensor._data.equal(torch.tensor([104, 105, 195, 169]))
    assert tensor._offset.equal(torch.tensor([0, 2, 4, 4]))

    tensor = StringTensor.from_arrow(pa.array([], type=pa.string()))
    assert tensor.size() == (0,)
    assert tensor.stride() == (1,)
    assert tensor.dtype == torch.uint8
    assert tensor._data.equal(torch.tensor([]))
    assert tensor._offset.equal(torch.tensor([0]))

    tensor = StringTensor.from_list([["hi", "é"], ["", "abc"]])
    array = tensor.to_arrow()
    assert array.type == pa.large_string()
    assert array.to_pylist() == ["hi", "é", "", "abc"]

    tensor = StringTensor.from_list(["hi", "é", "", "abc"])
    array = cast(StringTensor, tensor[1:]).to_arrow()
    assert array.offset == 1
    assert array.to_pylist() == ["é", "", "abc"]
    assert (
        array.buffers()[1].address
        == tensor._offset.numpy().__array_interface__["data"][0]
    )
    assert (
        array.buffers()[2].address
        == tensor._data.numpy().__array_interface__["data"][0]
    )


def _import_cudf() -> Any:
    pytest.importorskip("cupy")
    cudf = pytest.importorskip("cudf")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    return cudf


@withCUDA
def test_from_cudf(device: torch.device) -> None:
    cudf = _import_cudf()
    tensor = StringTensor.from_cudf(
        cudf.Series(["hi", "é", ""]),
        device=device,
    )

    assert tensor.device == device
    assert tensor.tolist() == ["hi", "é", ""]
    assert tensor._data.cpu().equal(
        torch.tensor([104, 105, 195, 169], dtype=torch.uint8)
    )
    assert tensor._offset.cpu().equal(torch.tensor([0, 2, 4, 4]))


@withCUDA
def test_from_cudf_sliced_values(device: torch.device) -> None:
    cudf = _import_cudf()
    tensor = StringTensor.from_cudf(
        cudf.Series(["x", "hi", "é", ""])[1:],
        device=device,
    )

    assert tensor.device == device
    assert tensor.tolist() == ["hi", "é", ""]


@withCUDA
def test_from_cudf_empty_values(device: torch.device) -> None:
    cudf = _import_cudf()
    tensor = StringTensor.from_cudf(
        cudf.Series([], dtype="object"),
        device=device,
    )

    assert tensor.device == device
    assert tensor.tolist() == []
    assert tensor._data.numel() == 0
    assert tensor._offset.cpu().equal(torch.tensor([0], dtype=torch.int32))


def test_from_cudf_errors() -> None:
    cudf = _import_cudf()

    with pytest.raises(ValueError, match="cannot represent null"):
        StringTensor.from_cudf(cudf.Series(["hi", None]))

    with pytest.raises(TypeError, match="string type"):
        StringTensor.from_cudf(cudf.Series([1, 2], dtype="int32"))


def test_allowed_dtype() -> None:
    tensor = StringTensor.from_list("hi")

    with pytest.raises(TypeError, match="Can't convert"):
        tensor.to(torch.float32)


def test_item() -> None:
    assert StringTensor.from_list("é").item() == "é"
    assert StringTensor.from_list(["hi", "é"])[1].item() == "é"
    assert str(StringTensor.from_list([""])) == ""

    with pytest.raises(RuntimeError, match="cannot be converted"):
        StringTensor.from_list(["hi", "é"]).item()


def test_tolist() -> None:
    for strings in [
        "é",
        ["hi", "é"],
        [],
        [["a", "bb", "c"], ["dd", "e", "ff"]],
    ]:
        assert StringTensor.from_list(strings).tolist() == strings
