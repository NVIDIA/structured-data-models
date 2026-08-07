import torch

from sdm import VarLenTensor

aten = torch.ops.aten


def test_stack_default_dimension() -> None:
    first = VarLenTensor.from_list([[0], [1, 2]])
    second = VarLenTensor.from_list([[3], None])
    out = aten.stack.default([first, second])

    assert isinstance(out, VarLenTensor)
    assert out.size() == (2, 2)
    assert out.is_contiguous()
    assert out.tolist() == [[[0], [1, 2]], [[3], None]]
