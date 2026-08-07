from collections.abc import Callable

import pytest
import torch
from torch import Tensor

from sdm import TableTensor

aten = torch.ops.aten


def make_table() -> TableTensor:
    return TableTensor(
        columns={"numerical": ("first", "second")},
        numerical=torch.arange(48).view(2, 3, 4, 2),
    )


def assert_matches_dense(out: TableTensor, expected: Tensor) -> None:
    assert type(out) is TableTensor
    assert out.size() == expected.size()
    assert out.stride() == expected.stride()
    assert out.storage_offset() == expected.storage_offset()
    assert out.numerical.equal(expected)


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
    op: Callable[[TableTensor], TableTensor],
    reference_op: Callable[[Tensor], Tensor],
) -> None:
    inp = make_table()

    out = op(inp)
    expected = reference_op(inp.numerical)

    assert_matches_dense(out, expected)
    assert out.columns == inp.columns
    assert (
        out.numerical.untyped_storage().data_ptr()
        == inp.numerical.untyped_storage().data_ptr()
    )
