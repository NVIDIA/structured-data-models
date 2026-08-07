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


def test_unbind_overload() -> None:
    base = make_columnar()
    inp = cast(ColumnarTensor, base.transpose(0, 1))
    expected = aten.unbind.int(dense(base).transpose(0, 1), 0)

    out = aten.unbind.int(inp, 0)

    assert isinstance(out, list)
    assert isinstance(inp.unbind(0), tuple)
    assert len(out) == len(expected)
    for actual, reference in zip(out, expected):
        assert_matches_dense(actual, reference)
        assert_shares_column_storage(actual, inp)


def test_split_overloads() -> None:
    base = make_columnar()
    inp = cast(ColumnarTensor, base.transpose(0, 1))
    reference = dense(base).transpose(0, 1)
    outputs = (
        aten.split.Tensor(inp, 2, 0),
        aten.split.sizes(inp, (1, 2), 0),
        aten.split.default(inp, (1, 2), 0),
        aten.split_with_sizes.default(inp, (1, 2), 0),
    )
    expected = (
        aten.split.Tensor(reference, 2, 0),
        aten.split.sizes(reference, (1, 2), 0),
        aten.split.default(reference, (1, 2), 0),
        aten.split_with_sizes.default(reference, (1, 2), 0),
    )

    for chunks, dense_chunks in zip(outputs, expected):
        assert isinstance(chunks, list)
        assert len(chunks) == len(dense_chunks)
        for chunk, dense_chunk in zip(chunks, dense_chunks):
            assert_matches_dense(chunk, dense_chunk)
            assert_shares_column_storage(chunk, inp)
    assert isinstance(inp.split(2, 0), tuple)


@pytest.mark.parametrize("split_size", [0, 1])
def test_split_empty_dimension(split_size: int) -> None:
    inp = ColumnarTensor((torch.empty(0, 3), torch.empty(0, 3)))
    expected = aten.split.Tensor(torch.empty(inp.size()), split_size, 0)

    out = aten.split.Tensor(inp, split_size, 0)

    assert len(out) == 1
    assert_matches_dense(out[0], expected[0])


@pytest.mark.parametrize("split_size", [-1, 0])
def test_split_size_validation(split_size: int) -> None:
    with pytest.raises(RuntimeError):
        aten.split.Tensor(make_columnar(), split_size, 0)


def test_column_dimension_unbind_and_split() -> None:
    inp = make_columnar()
    reference = dense(inp)

    columns = aten.unbind.int(inp, -1)
    chunks = aten.split.Tensor(inp, 1, -1)

    assert all(
        column.equal(expected)
        for column, expected in zip(columns, reference.unbind(-1))
    )
    assert all(
        column.untyped_storage().data_ptr()
        == source.untyped_storage().data_ptr()
        for column, source in zip(columns, inp.unbind(-1))
    )
    for chunk, expected in zip(chunks, reference.split(1, dim=-1)):
        assert_matches_dense(chunk, expected)
    for chunk, source in zip(chunks, inp.unbind(-1)):
        assert (
            chunk.unbind(-1)[0].untyped_storage().data_ptr()
            == source.untyped_storage().data_ptr()
        )
