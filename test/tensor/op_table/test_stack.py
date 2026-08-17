import torch
from torch import Tensor

from sdm import TableTensor

aten = torch.ops.aten


def make_table(offset: int = 0) -> TableTensor:
    return TableTensor(
        columns={"numerical": ("first", "second")},
        numerical=torch.arange(offset, offset + 48).view(2, 3, 4, 2),
    )


def assert_matches_dense(out: TableTensor, expected: Tensor) -> None:
    assert type(out) is TableTensor
    assert out.size() == expected.size()
    assert out.stride() == expected.stride()
    assert out.storage_offset() == expected.storage_offset()
    assert out.numerical.equal(expected)


def test_stack_preserves_layout() -> None:
    first = make_table()
    second = make_table(1000)

    out = aten.stack.default((first, second), 1)

    assert_matches_dense(
        out,
        aten.stack.default((first.numerical, second.numerical), 1),
    )
    assert out.columns == first.columns
