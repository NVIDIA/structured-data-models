import pyarrow as pa
import pytest
import torch

from sdm import NullableIntTensor


def test_from_list() -> None:
    tensor = NullableIntTensor.from_list([[1, None], [3, 4]])

    assert tensor.size() == (2, 2)
    assert tensor.dtype == torch.int64
    assert tensor.valid is not None
    assert tensor.tolist() == [[1, None], [3, 4]]

    tensor = NullableIntTensor.from_list(None)
    assert tensor.size() == ()
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


def test_validation_and_storage_helpers() -> None:
    with pytest.raises(ValueError, match="integer dtype"):
        NullableIntTensor(torch.tensor([1.0]))

    with pytest.raises(ValueError, match="have size"):
        NullableIntTensor(
            torch.tensor([1, 2]),
            valid=torch.tensor([True]),
        )

    tensor = NullableIntTensor.from_list([1, None])
    assert tensor.is_contiguous()
    assert tensor.contiguous() is tensor
    assert tensor.clone().tolist() == [1, None]
