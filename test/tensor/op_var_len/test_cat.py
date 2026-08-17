import pytest
import torch

from sdm import VarLenTensor

aten = torch.ops.aten


def test_cat_default_and_inner_dimension() -> None:
    top = VarLenTensor.from_list([[[0], [1]], [[2], None]])
    bottom = VarLenTensor.from_list([[[3], [4, 5]]])
    out = aten.cat.default([top, bottom])

    assert isinstance(out, VarLenTensor)
    assert out.size() == (3, 2)
    assert out.is_contiguous()
    assert out.tolist() == [
        [[0], [1]],
        [[2], None],
        [[3], [4, 5]],
    ]

    left = VarLenTensor.from_list([[[0]], [[1, 2]]])
    right = VarLenTensor.from_list([[[3], None], [[4], [5]]])
    inner = aten.cat.default([left, right], 1)
    assert inner.tolist() == [
        [[0], [3], None],
        [[1, 2], [4], [5]],
    ]


def test_cat_rejects_zero_dimensional_inputs() -> None:
    tensor = VarLenTensor.from_list([[0]]).squeeze()

    with pytest.raises(RuntimeError, match="zero-dimensional tensor"):
        aten.cat.default([tensor, tensor])


def test_cat_promotes_offset_dtype() -> None:
    int32 = VarLenTensor.from_list(
        [[0]],
        offset_dtype=torch.int32,
    )
    int64 = VarLenTensor.from_list(
        [[1]],
        offset_dtype=torch.int64,
    )

    out = aten.cat.default([int32, int64])

    assert out._offset.dtype == torch.int64
