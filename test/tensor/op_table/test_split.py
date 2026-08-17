from typing import cast

import pytest
import torch
from torch import Tensor

from sdm import Stype, TableTensor

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


def test_unbind_overload() -> None:
    inp = cast(TableTensor, make_table().transpose(0, 1))
    expected = aten.unbind.int(inp.numerical, 0)

    out = cast(list[TableTensor], aten.unbind.int(inp, 0))

    assert isinstance(out, list)
    assert isinstance(inp.unbind(0), tuple)
    assert len(out) == len(expected)
    for actual, reference in zip(out, expected):
        assert_matches_dense(actual, reference)
        assert actual.columns == inp.columns
        assert (
            actual.numerical.untyped_storage().data_ptr()
            == inp.numerical.untyped_storage().data_ptr()
        )


def test_split_overloads() -> None:
    inp = cast(TableTensor, make_table().transpose(0, 1))
    reference = inp.numerical
    outputs = (
        cast(list[TableTensor], aten.split.Tensor(inp, 2, 0)),
        cast(list[TableTensor], aten.split.sizes(inp, (1, 2), 0)),
        cast(list[TableTensor], aten.split.default(inp, (1, 2), 0)),
        cast(
            list[TableTensor],
            aten.split_with_sizes.default(inp, (1, 2), 0),
        ),
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
            assert chunk.columns == inp.columns
            assert (
                chunk.numerical.untyped_storage().data_ptr()
                == inp.numerical.untyped_storage().data_ptr()
            )
    assert isinstance(inp.split(2, 0), tuple)


def test_column_dimension_split_policy() -> None:
    inp = make_table()
    chunks = aten.split.Tensor(inp, 1, -1)
    unbound = aten.unbind.int(inp, -1)

    assert len(chunks) == inp.size(-1)
    assert len(unbound) == inp.size(-1)
    for i, (chunk, column) in enumerate(zip(chunks, unbound)):
        expected = inp.numerical[..., i : i + 1]
        assert_matches_dense(chunk, expected)
        assert_matches_dense(column, expected)
        name = inp.columns[Stype.numerical][i]
        assert chunk.columns[Stype.numerical] == (name,)
        assert column.columns[Stype.numerical] == (name,)
        assert (
            chunk.numerical.untyped_storage().data_ptr()
            == inp.numerical.untyped_storage().data_ptr()
        )
        assert (
            column.numerical.untyped_storage().data_ptr()
            == inp.numerical.untyped_storage().data_ptr()
        )

    with pytest.raises(RuntimeError, match="Can't split"):
        aten.split_with_sizes.default(inp, (1, 1), -1)


def test_empty_column_dimension_split_and_unbind() -> None:
    inp = TableTensor(size=(2, 3))

    chunks = aten.split.Tensor(inp, 1, -1)
    unbound = aten.unbind.int(inp, -1)

    assert isinstance(chunks, list)
    assert len(chunks) == 1
    assert type(chunks[0]) is TableTensor
    assert chunks[0].size() == inp.size()
    assert chunks[0].stride() == inp.stride()
    assert chunks[0].storage_offset() == inp.storage_offset()
    assert chunks[0].columns == inp.columns
    assert isinstance(unbound, list)
    assert unbound == []
