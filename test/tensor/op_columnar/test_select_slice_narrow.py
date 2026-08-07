from collections.abc import Callable
from typing import cast

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
            lambda inp: aten.select.int(inp, 1, 1),
            lambda inp: aten.select.int(inp, 1, 1),
            id="select.int",
        ),
        pytest.param(
            lambda inp: aten.slice.Tensor(inp, 1, 0, 3, 2),
            lambda inp: aten.slice.Tensor(inp, 1, 0, 3, 2),
            id="slice.Tensor",
        ),
        pytest.param(
            lambda inp: aten.narrow.default(inp, 1, 1, 2),
            lambda inp: aten.narrow.default(inp, 1, 1, 2),
            id="narrow.default",
        ),
    ],
)
def test_select_slice_narrow_overloads(
    op: Callable[[ColumnarTensor], ColumnarTensor],
    reference_op: Callable[[Tensor], Tensor],
) -> None:
    inp = make_columnar()

    out = op(inp)
    expected = reference_op(dense(inp))

    assert_matches_dense(out, expected)
    assert_shares_column_storage(out, inp)


def test_column_dimension_selection_and_views() -> None:
    inp = make_columnar()
    reference = dense(inp)

    selected = aten.select.int(inp, -1, 1)
    sliced = aten.slice.Tensor(inp, -1, 0, 2, 2)
    narrowed = aten.narrow.default(inp, -1, -1, 1)

    assert selected.equal(reference.select(-1, 1))
    assert (
        selected.untyped_storage().data_ptr()
        == inp.unbind(-1)[1].untyped_storage().data_ptr()
    )
    assert_matches_dense(sliced, reference[..., ::2])
    assert_matches_dense(narrowed, reference[..., -1:])
    assert_shares_column_storage(
        sliced,
        cast(ColumnarTensor, inp[..., ::2]),
    )
    assert (
        narrowed.unbind(-1)[0].untyped_storage().data_ptr()
        == inp.unbind(-1)[-1].untyped_storage().data_ptr()
    )
