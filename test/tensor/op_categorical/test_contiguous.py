import torch
from torch import Tensor

from sdm import CategoricalTensor

aten = torch.ops.aten


def _is_alias_of(left: Tensor, right: Tensor) -> bool:
    return torch._C._is_alias_of(  # ty: ignore[unresolved-attribute]
        left,
        right,
    )


def test_contiguous_returns_self_or_materializes_code() -> None:
    tensor = CategoricalTensor(
        code=torch.tensor(
            [[0, 1], [1, 0]],
            dtype=torch.int32,
        ),
        categories=(torch.tensor([0, 1]), torch.tensor([10, 20])),
    )
    assert aten.contiguous.default(tensor) is tensor

    transposed = CategoricalTensor(
        code=tensor.code.transpose(0, 1),
        categories=tensor.categories,
    )
    out = aten.contiguous.default(transposed)

    assert isinstance(out, CategoricalTensor)
    assert out is not transposed
    assert out.is_contiguous()
    assert out.tolist() == transposed.tolist()
    assert not _is_alias_of(out.code, transposed.code)
    assert all(
        actual.equal(expected)
        for actual, expected in zip(out.categories, transposed.categories)
    )
