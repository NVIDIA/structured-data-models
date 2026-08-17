import torch

from sdm import ColumnarTensor

aten = torch.ops.aten


def test_equality_overloads() -> None:
    inp = ColumnarTensor(
        (
            torch.arange(24.0).view(2, 3, 4),
            torch.arange(100.0, 124.0).view(2, 3, 4),
        ),
    )
    same = inp.clone()
    close = ColumnarTensor(
        (
            torch.arange(24.0).view(2, 3, 4) + 1e-7,
            torch.arange(100.0, 124.0).view(2, 3, 4) + 1e-7,
        ),
    )

    assert aten.equal.default(inp, same)
    assert aten.allclose.default(inp, close, 1e-5, 1e-6)
    assert not aten.equal.default(inp, close)
