import pyarrow as pa
import pytest
import torch

from sdm.tensor.io import to_arrow, to_cudf
from sdm.tensor.io.arrow import _combine_arrow_chunks
from sdm.testing import onlyCUDA


def test_combine_arrow_dictionary_string_chunks() -> None:
    array = pa.chunked_array(
        [
            pa.DictionaryArray.from_arrays([0, 1], ["b", "a"]),
            pa.DictionaryArray.from_arrays([0, 1], ["a", "c"]),
        ]
    )

    out = _combine_arrow_chunks(array)

    assert pa.types.is_dictionary(out.type)
    assert out.dictionary.type == pa.large_string()
    assert out.to_pylist() == ["b", "a", "a", "c"]


def test_to_arrow_bool() -> None:
    array = to_arrow(
        torch.tensor([True, False]),
        torch.tensor([True, False]),
    )

    assert array.type == pa.bool_()
    assert array.to_pylist() == [True, None]


@onlyCUDA
def test_to_cudf() -> None:
    pytest.importorskip("cudf")
    tensor = torch.arange(6, device="cuda").view(2, 3)

    series = to_cudf(tensor)
    output = torch.as_tensor(series)

    assert output.equal(tensor.view(-1))
    assert output.data_ptr() == tensor.data_ptr()


@onlyCUDA
def test_to_cudf_noncontiguous() -> None:
    pytest.importorskip("cudf")
    tensor = torch.arange(12, device="cuda").view(3, 4).t()
    assert not tensor.is_contiguous()

    series = to_cudf(tensor)

    assert torch.as_tensor(series).equal(tensor.contiguous().view(-1))


@onlyCUDA
def test_to_cudf_valid_mask() -> None:
    pytest.importorskip("cudf")
    tensor = torch.arange(3, device="cuda")
    valid_mask = torch.tensor([True, False, True], device="cuda")

    ser = to_cudf(tensor, valid_mask)

    assert ser.null_count == 1
    assert ser.to_arrow().to_pylist() == [0, None, 2]


@onlyCUDA
def test_to_cudf_requires_cuda() -> None:
    pytest.importorskip("cudf")
    with pytest.raises(ValueError, match="on a CUDA device"):
        to_cudf(torch.arange(3))
