import math

import pyarrow as pa
import pytest
import torch

from sdm.tensor.io import arrow_as_tensor, to_arrow, to_cudf
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


def test_arrow_bool() -> None:
    tensor = arrow_as_tensor(pa.array([True, False, True], type=pa.bool_()))
    assert tensor.dtype == torch.bool
    assert tensor.equal(torch.tensor([True, False, True]))

    assert to_arrow(tensor).to_pylist() == [True, False, True]

    valid_mask = torch.tensor([True, False, True])
    assert to_arrow(tensor, valid_mask).to_pylist() == [True, None, True]


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
    tensor = torch.tensor(
        [0.0, float("nan"), float("nan"), 3.0],
        device="cuda",
    )
    valid_mask = torch.tensor(
        [True, True, False, False],
        device="cuda",
    )

    ser = to_cudf(tensor, valid_mask)

    values = ser.to_arrow().to_pylist()
    assert ser.null_count == 2
    assert values == pytest.approx([0.0, math.nan, None, None], nan_ok=True)


@onlyCUDA
def test_to_cudf_requires_cuda() -> None:
    pytest.importorskip("cudf")
    with pytest.raises(ValueError, match="on a CUDA device"):
        to_cudf(torch.arange(3))
