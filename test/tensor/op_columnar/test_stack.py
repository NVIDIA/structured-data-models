import torch
from torch import Tensor

from sdm import ColumnarTensor

aten = torch.ops.aten


def make_columnar(offset: int = 0) -> ColumnarTensor:
    return ColumnarTensor(
        (
            torch.arange(offset, offset + 24).view(2, 3, 4),
            torch.arange(offset + 100, offset + 124).view(2, 3, 4),
        ),
    )


def dense(inp: ColumnarTensor) -> Tensor:
    return torch.tensor(inp.tolist())


def assert_matches_dense(out: ColumnarTensor, expected: Tensor) -> None:
    assert type(out) is ColumnarTensor
    assert out.size() == expected.size()
    assert out.stride() == expected.stride()
    assert out.storage_offset() == expected.storage_offset()
    assert out.tolist() == expected.tolist()


def test_stack_overload() -> None:
    first = make_columnar()
    second = make_columnar(1000)
    dense_inputs = (dense(first), dense(second))

    stack = aten.stack.default((first, second), 1)

    assert_matches_dense(stack, aten.stack.default(dense_inputs, 1))
