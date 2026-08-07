import pytest
import torch

from sdm import ColumnarTensor

aten = torch.ops.aten


def make_columnar() -> ColumnarTensor:
    return ColumnarTensor(
        (
            torch.arange(24.0).view(2, 3, 4),
            torch.arange(100.0, 124.0).view(2, 3, 4),
        ),
    )


def test_pin_memory_overloads() -> None:
    inp = make_columnar()

    assert not aten.is_pinned.default(inp)
    assert not aten.is_pinned.default(inp, torch.device("cpu"))
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError):
            aten._pin_memory.default(inp)
        with pytest.raises(RuntimeError):
            aten.pin_memory.default(inp)
        with pytest.raises(RuntimeError):
            aten._pin_memory.default(inp, torch.device("cpu"))
        with pytest.raises(RuntimeError):
            aten.pin_memory.default(inp, torch.device("cpu"))
        return

    pinned = aten._pin_memory.default(inp)
    composite = aten.pin_memory.default(inp)
    assert aten.is_pinned.default(pinned)
    assert aten.is_pinned.default(composite)
