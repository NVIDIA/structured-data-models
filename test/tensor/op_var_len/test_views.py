from collections.abc import Callable, Sequence
from typing import cast

import pytest
import torch
from torch import Tensor

from sdm import VarLenTensor

aten = torch.ops.aten


def _is_alias(left: Tensor, right: Tensor) -> bool:
    return torch._C._is_alias_of(  # ty: ignore[unresolved-attribute]
        left,
        right,
    )


def _scalars(size: Sequence[int]) -> tuple[VarLenTensor, Tensor]:
    dense = torch.arange(torch.Size(size).numel()).view(tuple(size))
    return VarLenTensor.from_tensor(dense), dense


def _assert_matches_dense(out: VarLenTensor, dense: Tensor) -> None:
    assert isinstance(out, VarLenTensor)
    assert out.size() == dense.size()
    assert out.stride() == dense.stride()
    assert out.storage_offset() == dense.storage_offset()
    assert out.tolist() == VarLenTensor.from_tensor(dense).tolist()


@pytest.mark.parametrize(
    "operation",
    [
        pytest.param(
            lambda tensor: aten.view.default(tensor, [3, 2]),
            id="view.default",
        ),
        pytest.param(
            lambda tensor: aten.reshape.default(tensor, [3, 2]),
            id="reshape.default",
        ),
        pytest.param(
            lambda tensor: aten.flatten.using_ints(tensor, 0, -1),
            id="flatten.using_ints",
        ),
    ],
)
def test_contiguous_shape_operations(
    operation: Callable[[Tensor], Tensor],
) -> None:
    tensor, dense = _scalars((2, 3))
    out = operation(tensor)
    expected = operation(dense)

    _assert_matches_dense(cast(VarLenTensor, out), expected)
    assert _is_alias(out, tensor)


def test_unsafe_view_preserves_inner_storage_without_outer_alias() -> None:
    tensor, dense = _scalars((2, 3))
    out = aten._unsafe_view.default(tensor, [3, 2])
    expected = aten._unsafe_view.default(dense, [3, 2])

    _assert_matches_dense(out, expected)
    assert not _is_alias(out, tensor)
    assert out._data.data_ptr() == tensor._data.data_ptr()
    assert out._offset.data_ptr() == tensor._offset.data_ptr()


def test_reshape_materializes_incompatible_layout() -> None:
    dense = torch.arange(9).as_strided((3, 2), (3, 1))
    tensor = VarLenTensor.from_tensor(dense)
    out = aten.reshape.default(tensor, [2, 3])
    expected = aten.reshape.default(dense, [2, 3])

    _assert_matches_dense(out, expected)
    assert not _is_alias(out, tensor)


def test_squeeze_overloads() -> None:
    tensor, dense = _scalars((1, 2, 1, 3))
    cases = (
        (
            aten.squeeze.default(tensor),
            aten.squeeze.default(dense),
        ),
        (
            aten.squeeze.dim(tensor, 0),
            aten.squeeze.dim(dense, 0),
        ),
        (
            aten.squeeze.dims(tensor, [0, 2]),
            aten.squeeze.dims(dense, [0, 2]),
        ),
    )

    for out, expected in cases:
        _assert_matches_dense(out, expected)
        assert _is_alias(out, tensor)


def test_unsqueeze_and_expand() -> None:
    tensor, dense = _scalars((1, 3))
    unsqueezed = aten.unsqueeze.default(tensor, 0)
    expected_unsqueezed = aten.unsqueeze.default(dense, 0)
    expanded = aten.expand.default(tensor, [2, 3], implicit=True)
    expected_expanded = aten.expand.default(dense, [2, 3], implicit=True)

    _assert_matches_dense(unsqueezed, expected_unsqueezed)
    _assert_matches_dense(expanded, expected_expanded)
    assert _is_alias(unsqueezed, tensor)
    assert _is_alias(expanded, tensor)


def test_dimension_reordering_overloads() -> None:
    tensor, dense = _scalars((2, 3))
    cases = (
        (aten.t.default(tensor), aten.t.default(dense)),
        (
            aten.transpose.int(tensor, 0, 1),
            aten.transpose.int(dense, 0, 1),
        ),
        (
            aten.permute.default(tensor, [1, 0]),
            aten.permute.default(dense, [1, 0]),
        ),
    )

    for out, expected in cases:
        _assert_matches_dense(out, expected)
        assert _is_alias(out, tensor)


def test_basic_index_view_overloads() -> None:
    tensor, dense = _scalars((2, 3, 2))
    cases = (
        (
            aten.select.int(tensor, 1, 1),
            aten.select.int(dense, 1, 1),
        ),
        (
            aten.slice.Tensor(tensor, 1, 0, 3, 2),
            aten.slice.Tensor(dense, 1, 0, 3, 2),
        ),
        (
            aten.narrow.default(tensor, 1, 1, 2),
            aten.narrow.default(dense, 1, 1, 2),
        ),
    )

    for out, expected in cases:
        _assert_matches_dense(out, expected)
        assert _is_alias(out, tensor)


def test_chained_views_share_the_ultimate_logical_base() -> None:
    tensor = VarLenTensor.from_list(
        [[[0], [1]], [[2], None]],
    )
    first = tensor.view(2, 2)
    second = cast(VarLenTensor, first[:1])

    assert first._base is tensor
    assert second._base is tensor
    assert _is_alias(second, tensor)
    assert second._data is tensor._data
    assert second._offset is tensor._offset
    assert second._valid is tensor._valid

    second._data[0] = 9
    valid = second.valid
    assert valid is not None
    valid[0, 1] = False
    assert tensor.tolist() == [[[9], None], [[2], None]]
