from typing import cast

import torch
from torch import Tensor

from sdm import VarLenTensor

aten = torch.ops.aten


def _is_alias(left: Tensor, right: Tensor) -> bool:
    return torch._C._is_alias_of(  # ty: ignore[unresolved-attribute]
        left,
        right,
    )


def _assert_matches_dense(out: VarLenTensor, dense: Tensor) -> None:
    assert isinstance(out, VarLenTensor)
    assert out.size() == dense.size()
    assert out.stride() == dense.stride()
    assert out.storage_offset() == dense.storage_offset()
    assert out.tolist() == VarLenTensor.from_tensor(dense).tolist()


def test_unbind_exact_and_public_return_structures() -> None:
    dense = torch.arange(6).view(2, 3)
    tensor = VarLenTensor.from_tensor(dense)
    exact = aten.unbind.int(tensor, 1)
    expected = aten.unbind.int(dense, 1)
    public = tensor.unbind(1)

    assert isinstance(exact, list)
    assert isinstance(public, tuple)
    assert len(exact) == len(expected) == len(public)
    for out, expected_out in zip(exact, expected):
        _assert_matches_dense(
            cast(VarLenTensor, out),
            cast(Tensor, expected_out),
        )
        assert _is_alias(cast(Tensor, out), tensor)


def test_split_overloads_and_public_return_structure() -> None:
    dense = torch.arange(8).view(2, 4)
    tensor = VarLenTensor.from_tensor(dense)
    cases = (
        (
            aten.split.Tensor(tensor, 2, 1),
            aten.split.Tensor(dense, 2, 1),
        ),
        (
            aten.split.sizes(tensor, [1, 3], 1),
            aten.split.sizes(dense, [1, 3], 1),
        ),
        (
            aten.split.default(tensor, [1, 3], 1),
            aten.split.default(dense, [1, 3], 1),
        ),
        (
            aten.split_with_sizes.default(tensor, [1, 3], 1),
            aten.split_with_sizes.default(dense, [1, 3], 1),
        ),
    )

    assert isinstance(tensor.split(2, dim=1), tuple)
    for outputs, expected_outputs in cases:
        assert isinstance(outputs, list)
        assert len(outputs) == len(expected_outputs)
        for out, expected in zip(outputs, expected_outputs):
            _assert_matches_dense(
                cast(VarLenTensor, out),
                cast(Tensor, expected),
            )
            assert _is_alias(cast(Tensor, out), tensor)
