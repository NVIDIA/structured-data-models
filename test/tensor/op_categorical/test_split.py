from collections.abc import Sequence

import torch
from torch import Tensor

from sdm import CategoricalTensor

aten = torch.ops.aten


def _is_alias_of(left: Tensor, right: Tensor) -> bool:
    return torch._C._is_alias_of(  # ty: ignore[unresolved-attribute]
        left,
        right,
    )


def _tensor() -> CategoricalTensor:
    code = torch.arange(24, dtype=torch.int32).remainder(4).view(2, 3, 4)
    categories = tuple(torch.arange(4) + 10 * i for i in range(4))
    return CategoricalTensor(code=code, categories=categories)


def _assert_categories(
    actual: Sequence[Tensor],
    expected: Sequence[Tensor],
) -> None:
    assert len(actual) == len(expected)
    assert all(
        actual_category is expected_category
        for actual_category, expected_category in zip(actual, expected)
    )


def test_unbind_and_split_overloads_preserve_schema_aliases() -> None:
    tensor = _tensor()

    row_outputs = (
        aten.unbind.int(tensor, 0),
        aten.split.Tensor(tensor, 1),
        aten.split.sizes(tensor, (1, 1), 0),
        aten.split.default(tensor, (1, 1), 0),
        aten.split_with_sizes.default(tensor, (1, 1), 0),
    )
    assert all(isinstance(outputs, list) for outputs in row_outputs)
    for outputs in row_outputs:
        assert len(outputs) == 2
        for out in outputs:
            assert isinstance(out, CategoricalTensor)
            _assert_categories(out.categories, tensor.categories)
            assert _is_alias_of(out, tensor)

    column_outputs = aten.split.Tensor(tensor, 2, -1)
    assert len(column_outputs) == 2
    for i, out in enumerate(column_outputs):
        assert isinstance(out, CategoricalTensor)
        _assert_categories(
            out.categories, tensor.categories[2 * i : 2 * i + 2]
        )
        assert _is_alias_of(out, tensor)

    assert all(type(out) is Tensor for out in aten.unbind.int(tensor, -1))

    assert isinstance(torch.unbind(tensor, 0), tuple)
    assert isinstance(torch.split(tensor, 1, 0), tuple)
