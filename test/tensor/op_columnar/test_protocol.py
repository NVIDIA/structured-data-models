from typing import cast

import pytest
import torch

from sdm import ColumnarTensor


def make_columnar() -> ColumnarTensor:
    return ColumnarTensor(
        (
            torch.arange(24).view(2, 3, 4),
            torch.arange(100, 124).view(2, 3, 4),
        ),
    )


@pytest.mark.parametrize("inference", [False, True])
def test_tensor_flatten_round_trip(inference: bool) -> None:
    with torch.inference_mode(inference):
        inp = cast(ColumnarTensor, make_columnar().transpose(0, 1)[1:])

    attrs, ctx = inp.__tensor_flatten__()
    out = type(inp).__tensor_unflatten__(
        {name: getattr(inp, name) for name in attrs},
        ctx,
        inp.size(),
        inp.stride(),
    )

    assert type(out) is ColumnarTensor
    assert out.size() == inp.size()
    assert out.stride() == inp.stride()
    assert out.storage_offset() == inp.storage_offset()
    assert out.dtype == inp.dtype
    assert out.device == inp.device
    assert out.layout == inp.layout
    assert out.requires_grad == inp.requires_grad
    assert out.is_inference() == inp.is_inference()
    assert out.tolist() == inp.tolist()
