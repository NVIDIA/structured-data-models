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
def test_index_view_overloads(
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


def test_narrow_rejects_column_dimension() -> None:
    inp = make_table()

    with pytest.raises(RuntimeError, match="column dimension"):
        aten.narrow.default(inp, -1, 0, 1)


def test_negative_narrow_preserves_layout() -> None:
    inp = make_table()

    out = aten.narrow.default(inp, -1, -inp.size(-1), inp.size(-1))

    assert_matches_dense(out, inp.numerical)
    assert out.columns == inp.columns


def test_slice_preserves_input_inference_state() -> None:
    normal = make_table()
    with torch.inference_mode():
        normal_out = aten.slice.Tensor(normal, 1, 1, 3)
        inference = make_table()

    inference_out = aten.slice.Tensor(inference, 1, 1, 3)

    assert not normal_out.is_inference()
    assert inference_out.is_inference()
