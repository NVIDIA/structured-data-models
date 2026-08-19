import pyarrow as pa
import pytest
import torch

from sdm import NullableIntTensor
from sdm.testing import onlyCUDA


def test_from_list() -> None:
    tensor = NullableIntTensor.from_list([[1, None], [3, 4]])

    assert tensor.size() == (2, 2)
    assert tensor.dtype == torch.int64
    assert tensor.tolist() == [[1, None], [3, 4]]

    tensor = NullableIntTensor.from_list(None)
    assert tensor.size() == ()
    assert tensor.dtype == torch.int64
    assert tensor.tolist() is None
    assert tensor.item() is None


def test_arrow() -> None:
    tensor = NullableIntTensor.from_arrow(
        pa.array([1, None, 3], type=pa.int32()),
        size=(1, 3),
    )

    assert tensor.size() == (1, 3)
    assert tensor.dtype == torch.int32
    assert tensor.tolist() == [[1, None, 3]]
    assert tensor.to_arrow().to_pylist() == [1, None, 3]

    tensor = NullableIntTensor.from_arrow(pa.array([], type=pa.int32()))
    assert tensor.size() == (0,)


@onlyCUDA
def test_cudf() -> None:
    cudf = pytest.importorskip("cudf")

    ser = cudf.Series([1, None, 3], dtype="int32")
    tensor = NullableIntTensor.from_cudf(ser, size=(1, 3))

    assert tensor.size() == (1, 3)
    assert tensor.dtype == torch.int32
    assert tensor.tolist() == [[1, None, 3]]
    assert tensor.to_cudf().to_arrow().to_pylist() == [1, None, 3]

    tensor = NullableIntTensor.from_cudf(cudf.Series([], dtype="int32"))
    assert tensor.size() == (0,)


def test_nan_to_num() -> None:
    tensor = NullableIntTensor.from_list([1, None, 3])

    out = torch.nan_to_num(tensor)
    assert out.equal(torch.tensor([1, 0, 3]))

    out = torch.nan_to_num(tensor, nan=-1)
    assert out.equal(torch.tensor([1, -1, 3]))
