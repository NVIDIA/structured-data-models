from typing import cast

import pyarrow as pa
import pytest
import torch
from sdm import StringTensor
from sdm.testing import onlyCUDA, withCUDA


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


@onlyCUDA
def test_from_cudf() -> None:
    cudf = pytest.importorskip("cudf")

    tensor = StringTensor.from_cudf(
        cudf.Series(["hi", "é", ""]),
    )

    assert tensor.is_cuda
    assert tensor.tolist() == ["hi", "é", ""]
    assert tensor._data.equal(
        torch.tensor([104, 105, 195, 169], device=tensor.device)
    )
    assert tensor._offset.equal(
        torch.tensor([0, 2, 4, 4], device=tensor.device)
    )


@onlyCUDA
def test_from_cudf_sliced_values() -> None:
    cudf = pytest.importorskip("cudf")

    tensor = StringTensor.from_cudf(
        cudf.Series(["x", "hi", "é", ""])[1:],
    )

    assert tensor.is_cuda
    assert tensor.tolist() == ["hi", "é", ""]
    # The slice anchors into the base offsets instead of copying:
    assert int(tensor.storage_offset()) == 1
    assert tensor._offset.equal(
        torch.tensor([0, 1, 3, 5, 5], device=tensor.device)
    )


@onlyCUDA
def test_from_cudf_all_empty_values() -> None:
    cudf = pytest.importorskip("cudf")

    tensor = StringTensor.from_cudf(
        cudf.Series(["", ""]),
    )

    assert tensor.is_cuda
    assert tensor.tolist() == ["", ""]
    assert tensor._data.numel() == 0
    assert tensor._offset.equal(torch.tensor([0, 0, 0], device=tensor.device))


@onlyCUDA
def test_from_cudf_empty_values() -> None:
    cudf = pytest.importorskip("cudf")

    tensor = StringTensor.from_cudf(
        cudf.Series([], dtype="object"),
    )

    assert tensor.is_cuda
    assert tensor.tolist() == []
    assert tensor._data.numel() == 0
    assert tensor._offset.equal(torch.tensor([0], device=tensor.device))


@pytest.mark.parametrize(
    "offset_dtype",
    [torch.int32, torch.int64],
    ids=["int32", "int64"],
)
@onlyCUDA
def test_to_cudf(offset_dtype: torch.dtype) -> None:
    pytest.importorskip("cudf")
    values = ["é", "東京", "🙂", ""]
    tensor = StringTensor.from_list(
        values,
        device="cuda",
        offset_dtype=offset_dtype,
    )

    series = tensor.to_cudf()

    assert tensor._offset.dtype == offset_dtype
    assert series.to_arrow().to_pylist() == values


@pytest.mark.parametrize(
    "offset_dtype",
    [torch.int32, torch.int64],
    ids=["int32", "int64"],
)
@onlyCUDA
def test_to_cudf_empty(offset_dtype: torch.dtype) -> None:
    cudf = pytest.importorskip("cudf")
    tensor = StringTensor.from_list(
        [],
        device="cuda",
        offset_dtype=offset_dtype,
    )

    series = tensor.to_cudf()

    assert tensor._data.numel() == 0
    assert tensor._offset.equal(
        torch.zeros(1, dtype=offset_dtype, device=tensor.device)
    )
    assert cudf.api.types.is_string_dtype(series.dtype)
    assert series.to_arrow().to_pylist() == []


def test_to_cudf_requires_cuda() -> None:
    tensor = StringTensor.from_list(["a"])

    with pytest.raises(RuntimeError, match="on a CUDA device"):
        tensor.to_cudf()


@onlyCUDA
def test_from_cudf_errors() -> None:
    cudf = pytest.importorskip("cudf")

    with pytest.raises(ValueError, match="cannot represent null"):
        StringTensor.from_cudf(cudf.Series(["hi", None]))

    with pytest.raises(TypeError, match="string type"):
        StringTensor.from_cudf(cudf.Series([1, 2], dtype="int32"))


def test_allowed_dtype() -> None:
    tensor = StringTensor.from_list("hi")

    with pytest.raises(TypeError, match="Can't convert"):
        tensor.to(torch.float32)


def test_to_dtype_layout_copy() -> None:
    tensor = StringTensor.from_list(["hi", "é", ""])

    out = tensor.to(torch.uint8, copy=True)
    assert isinstance(out, StringTensor)
    assert out.tolist() == tensor.tolist()

    out = tensor.clone()
    assert isinstance(out, StringTensor)
    assert out.tolist() == tensor.tolist()


@pytest.mark.parametrize("create_in_inference_mode", [False, True])
def test_to_list_in_inference_mode(
    create_in_inference_mode: bool,
) -> None:
    values = ["hi", "é", ""]
    if create_in_inference_mode:
        with torch.inference_mode():
            tensor = StringTensor.from_list(values)
    else:
        tensor = StringTensor.from_list(values)

    with torch.inference_mode():
        assert tensor.tolist() == values


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


@withCUDA
def test_sort(device: torch.device) -> None:
    tensor = StringTensor.from_list(["b", "aa", "a", "é", ""], device=device)

    out, perm = tensor.sort()
    assert out.device == device
    assert out.tolist() == ["", "a", "aa", "b", "é"]
    assert perm.equal(torch.tensor([4, 2, 1, 0, 3], device=device))

    out, perm = torch.sort(tensor, dim=-1, descending=True)
    assert out.device == device
    assert out.tolist() == ["é", "b", "aa", "a", ""]
    assert perm.equal(torch.tensor([3, 0, 1, 2, 4], device=device))

    perm = torch.argsort(tensor)
    assert perm.equal(torch.tensor([4, 2, 1, 0, 3], device=device))


def test_character_ngrams() -> None:
    tensor = StringTensor.from_list(["cat", "hi cat"])

    flat, offset = tensor.character_ngrams((2, 2))

    assert flat.tolist() == [
        " c",
        "ca",
        "at",
        "t ",
        " h",
        "hi",
        "i ",
        " c",
        "ca",
        "at",
        "t ",
    ]
    assert offset.equal(torch.tensor([0, 4, 11]))


def test_character_ngrams_combines_sizes() -> None:
    tensor = StringTensor.from_list(["cat"])

    flat, offset = tensor.character_ngrams((2, 3))

    assert flat.tolist() == [
        " c",
        "ca",
        "at",
        "t ",
        " ca",
        "cat",
        "at ",
    ]
    assert offset.equal(torch.tensor([0, 7]))


def test_character_ngrams_short_word_counts_once() -> None:
    tensor = StringTensor.from_list(["a"])

    flat, offset = tensor.character_ngrams((5, 5))

    assert flat.tolist() == [" a "]
    assert offset.equal(torch.tensor([0, 1]))


def test_character_ngrams_empty_string_yields_nothing() -> None:
    tensor = StringTensor.from_list([""])

    flat, offset = tensor.character_ngrams((2, 2))

    assert flat.tolist() == []
    assert offset.equal(torch.tensor([0, 0]))


def test_character_ngrams_lowercases_by_default() -> None:
    tensor = StringTensor.from_list(["CAT"])

    flat, _ = tensor.character_ngrams((3, 3))
    assert flat.tolist() == [" ca", "cat", "at "]

    flat, _ = tensor.character_ngrams((3, 3), lowercase=False)
    assert flat.tolist() == [" CA", "CAT", "AT "]


def test_character_ngrams_rejects_multi_dimensional_input() -> None:
    tensor = StringTensor.from_list([["a", "b"], ["c", "d"]])

    with pytest.raises(NotImplementedError, match="one-dimensional"):
        tensor.character_ngrams((2, 2))
