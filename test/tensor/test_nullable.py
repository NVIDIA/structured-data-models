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


def test_as_strided() -> None:
    tensor = NullableIntTensor(
        data=torch.arange(6).view(2, 3),
        valid=torch.tensor([[True, False, True], [False, True, True]]),
    )

    out = torch.as_strided(tensor, size=(3,), stride=(1,), storage_offset=3)

    assert out.tolist() == [None, 4, 5]
    assert out.storage_offset() == 3

    mismatched = NullableIntTensor(
        data=torch.arange(6).view(2, 3),
        valid=torch.tensor(
            [[True, False], [False, True], [True, False]],
        ).t(),
    )
    out = torch.as_strided(
        mismatched,
        size=(3,),
        stride=(1,),
        storage_offset=3,
    )
    assert isinstance(out, NullableIntTensor)
    assert out.tolist() == [None, 4, None]
    assert out._valid.stride() == (2,)
    assert torch._C._is_alias_of(  # ty: ignore[unresolved-attribute]
        out,
        mismatched,
    )
    assert torch._C._is_alias_of(  # ty: ignore[unresolved-attribute]
        out._valid,
        mismatched._valid,
    )

    compiled = torch.compile(
        lambda value: value[-1],
        fullgraph=True,
        backend="aot_eager",
    )
    compiled_out = compiled(mismatched)
    assert isinstance(compiled_out, NullableIntTensor)
    assert compiled_out.tolist() == [None, 4, None]
    assert torch._C._is_alias_of(  # ty: ignore[unresolved-attribute]
        compiled_out,
        mismatched,
    )
