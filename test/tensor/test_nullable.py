import pyarrow as pa
import pytest
import torch

from sdm import NullableTensor
from sdm.testing import onlyCUDA


def test_from_list() -> None:
    tensor = NullableTensor.from_list([[1, None], [3, 4]])

    assert tensor.size() == (2, 2)
    assert tensor.dtype == torch.int64
    assert tensor.tolist() == [[1, None], [3, 4]]

    tensor = NullableTensor.from_list(None)
    assert tensor.size() == ()
    assert tensor.dtype == torch.int64
    assert tensor.tolist() is None
    assert tensor.item() is None


def test_arrow() -> None:
    tensor = NullableTensor.from_arrow(
        pa.array([1, None, 3], type=pa.int32()),
        size=(1, 3),
    )

    assert tensor.size() == (1, 3)
    assert tensor.dtype == torch.int32
    assert tensor.tolist() == [[1, None, 3]]
    assert tensor.to_arrow().to_pylist() == [1, None, 3]

    tensor = NullableTensor.from_arrow(pa.array([], type=pa.int32()))
    assert tensor.size() == (0,)


def test_bool() -> None:
    tensor = NullableTensor.from_list([True, None, False])
    assert tensor.dtype == torch.bool
    assert tensor.tolist() == [True, None, False]

    tensor = NullableTensor.from_arrow(
        pa.array([True, None, False], type=pa.bool_()),
        size=(1, 3),
    )
    assert tensor.to_arrow().type == pa.bool_()
    assert tensor.to_arrow().to_pylist() == [True, None, False]


@onlyCUDA
def test_cudf() -> None:
    cudf = pytest.importorskip("cudf")

    ser = cudf.Series([1, None, 3], dtype="int32")
    tensor = NullableTensor.from_cudf(ser, size=(1, 3))

    assert tensor.size() == (1, 3)
    assert tensor.dtype == torch.int32
    assert tensor.tolist() == [[1, None, 3]]
    assert tensor.to_cudf().to_arrow().to_pylist() == [1, None, 3]

    tensor = NullableTensor.from_cudf(cudf.Series([], dtype="int32"))
    assert tensor.size() == (0,)


@onlyCUDA
def test_cudf_bool() -> None:
    cudf = pytest.importorskip("cudf")

    ser = cudf.Series([True, None, False], dtype="bool")
    tensor = NullableTensor.from_cudf(ser, size=(1, 3))

    assert tensor.size() == (1, 3)
    assert tensor.dtype == torch.bool
    assert tensor.tolist() == [[True, None, False]]
    assert tensor.to_cudf().to_arrow().to_pylist() == [True, None, False]


def test_nan_to_num() -> None:
    tensor = NullableTensor.from_list([1, None, 3])

    out = torch.nan_to_num(tensor)
    assert out.equal(torch.tensor([1, 0, 3]))

    out = torch.nan_to_num(tensor, nan=-1)
    assert out.equal(torch.tensor([1, -1, 3]))


def test_isnan_isfinite() -> None:
    tensor = NullableTensor.from_list([1, None, 3])

    out = torch.isnan(tensor)
    assert out.dtype == torch.bool
    assert out.equal(torch.tensor([False, True, False]))

    out = tensor.isfinite()
    assert out.dtype == torch.bool
    assert out.equal(torch.tensor([True, False, True]))

    out[0] = False
    assert tensor.tolist() == [1, None, 3]
