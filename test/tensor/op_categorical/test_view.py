from collections.abc import Sequence

import pytest
import torch
from torch import Tensor

from sdm import CategoricalTensor

aten = torch.ops.aten


def _is_alias_of(left: Tensor, right: Tensor) -> bool:
    return torch._C._is_alias_of(  # ty: ignore[unresolved-attribute]
        left,
        right,
    )


def _tensor(*, inference: bool = False) -> CategoricalTensor:
    with torch.inference_mode(inference):
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


@pytest.mark.parametrize("inference", [False, True])
def test_view_overloads_preserve_categories_aliases_and_inference(
    inference: bool,
) -> None:
    tensor = _tensor(inference=inference)

    with torch.inference_mode(not inference):
        outputs = (
            (aten.alias.default(tensor), tensor.code, True),
            (aten.view.default(tensor, (6, 4)), tensor.code.view(6, 4), True),
            (
                aten._unsafe_view.default(tensor, (6, 4)),
                aten._unsafe_view.default(tensor.code, (6, 4)),
                False,
            ),
            (
                aten.reshape.default(tensor, (6, 4)),
                tensor.code.reshape(6, 4),
                True,
            ),
            (
                aten.flatten.using_ints(tensor, 0, 1),
                tensor.code.flatten(0, 1),
                True,
            ),
            (
                aten.unsqueeze.default(tensor, 0),
                tensor.code.unsqueeze(0),
                True,
            ),
            (
                aten.transpose.int(tensor, 0, 1),
                tensor.code.transpose(0, 1),
                True,
            ),
            (
                aten.permute.default(tensor, (1, 0, 2)),
                tensor.code.permute(1, 0, 2),
                True,
            ),
            (
                aten.select.int(tensor, 0, 1),
                tensor.code.select(0, 1),
                True,
            ),
            (
                aten.slice.Tensor(tensor, 1, 0, 2, 1),
                aten.slice.Tensor(tensor.code, 1, 0, 2, 1),
                True,
            ),
            (
                aten.narrow.default(tensor, 1, 0, 2),
                tensor.code.narrow(1, 0, 2),
                True,
            ),
        )

    for out, expected_code, aliases_outer in outputs:
        assert isinstance(out, CategoricalTensor)
        assert out.is_inference() == tensor.is_inference()
        assert out.code.equal(expected_code)
        _assert_categories(out.categories, tensor.categories)
        assert _is_alias_of(out.code, tensor.code)
        assert _is_alias_of(out, tensor) == aliases_outer


@pytest.mark.parametrize("inference", [False, True])
def test_squeeze_and_expand_overloads_preserve_inference(
    inference: bool,
) -> None:
    tensor = _tensor(inference=inference)
    with torch.inference_mode(inference):
        singleton = CategoricalTensor(
            code=tensor.code.unsqueeze(0),
            categories=tensor.categories,
        )
        expandable = CategoricalTensor(
            code=tensor.code[:1],
            categories=tensor.categories,
        )

    with torch.inference_mode(not inference):
        outputs = (
            aten.squeeze.default(singleton),
            aten.squeeze.dim(singleton, 0),
            aten.squeeze.dims(singleton, (0,)),
            aten.expand.default(expandable, (2, 3, 4)),
        )

    for out in outputs:
        assert isinstance(out, CategoricalTensor)
        assert out.is_inference() == tensor.is_inference()
        assert out.size() == tensor.size()
        _assert_categories(out.categories, tensor.categories)
        assert _is_alias_of(out, singleton) or _is_alias_of(out, expandable)


def test_column_views_transform_category_vectors() -> None:
    tensor = _tensor()

    sliced = aten.slice.Tensor(tensor, -1, 1, 4, 2)
    assert isinstance(sliced, CategoricalTensor)
    _assert_categories(sliced.categories, tensor.categories[1::2])
    assert _is_alias_of(sliced, tensor)

    narrowed = aten.narrow.default(tensor, -1, 1, 2)
    assert isinstance(narrowed, CategoricalTensor)
    _assert_categories(narrowed.categories, tensor.categories[1:3])
    assert _is_alias_of(narrowed, tensor)


def test_views_materialize_when_the_final_axis_loses_column_semantics() -> (
    None
):
    tensor = _tensor()

    outputs = (
        aten.view.default(tensor, (-1,)),
        aten.flatten.using_ints(tensor, 1, 2),
        aten.unsqueeze.default(tensor, -1),
        aten.transpose.int(tensor, 0, -1),
        aten.permute.default(tensor, (2, 0, 1)),
        aten.select.int(tensor, -1, 0),
        aten.as_strided.default(tensor, (2, 3), (12, 4), 0),
    )

    assert all(type(out) is Tensor for out in outputs)

    narrowed = aten.narrow.Tensor(tensor, 0, torch.tensor(0), 1)
    assert isinstance(narrowed, CategoricalTensor)
    _assert_categories(narrowed.categories, tensor.categories)
    assert _is_alias_of(narrowed, tensor)
