from collections.abc import Callable

import pytest
import torch

from sdm import VarLenTensor

aten = torch.ops.aten


def _tensor() -> VarLenTensor:
    return VarLenTensor.from_list([[1.0], None, [2.0, 3.0]])


def test_is_pinned_accepts_explicit_device() -> None:
    tensor = _tensor()

    assert not aten.is_pinned.default(tensor)
    assert not aten.is_pinned.default(tensor, torch.device("cuda"))


@pytest.mark.parametrize(
    "operation",
    [
        pytest.param(aten._pin_memory.default, id="_pin_memory.default"),
        pytest.param(aten.pin_memory.default, id="pin_memory.default"),
    ],
)
def test_pin_memory_overloads(
    operation: Callable[[VarLenTensor], VarLenTensor],
) -> None:
    tensor = _tensor()
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError):
            operation(tensor)
        return

    out = operation(tensor)
    assert isinstance(out, VarLenTensor)
    assert out.is_pinned()
    assert out.tolist() == tensor.tolist()
