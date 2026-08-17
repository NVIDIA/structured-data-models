from collections.abc import Callable

import pytest
import torch
from torch import Tensor

from sdm import ColumnarTensor

aten = torch.ops.aten


def make_columnar() -> ColumnarTensor:
    return ColumnarTensor(
        (
            torch.arange(24).view(2, 3, 4),
            torch.arange(100, 124).view(2, 3, 4),
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


def assert_shares_column_storage(
    out: ColumnarTensor,
    inp: ColumnarTensor,
) -> None:
    for actual, source in zip(out.unbind(-1), inp.unbind(-1)):
        assert (
            actual.untyped_storage().data_ptr()
            == source.untyped_storage().data_ptr()
        )


@pytest.mark.parametrize(
    ("op", "reference_op"),
    [
        pytest.param(
            lambda inp: aten.unsqueeze.default(inp, 0),
            lambda inp: aten.unsqueeze.default(inp, 0),
            id="unsqueeze.default",
        ),
        pytest.param(
            lambda inp: aten.expand.default(inp[:1], (3, 3, 4, 2)),
            lambda inp: aten.expand.default(inp[:1], (3, 3, 4, 2)),
            id="expand.default",
        ),
        pytest.param(
            lambda inp: aten.transpose.int(inp, 0, 1),
            lambda inp: aten.transpose.int(inp, 0, 1),
            id="transpose.int",
        ),
        pytest.param(
            lambda inp: aten.permute.default(inp, (1, 0, 2, 3)),
            lambda inp: aten.permute.default(inp, (1, 0, 2, 3)),
            id="permute.default",
        ),
    ],
)
def test_dimension_view_overloads(
    op: Callable[[ColumnarTensor], ColumnarTensor],
    reference_op: Callable[[Tensor], Tensor],
) -> None:
    inp = make_columnar()

    out = op(inp)
    expected = reference_op(dense(inp))

    assert_matches_dense(out, expected)
    assert_shares_column_storage(out, inp)
