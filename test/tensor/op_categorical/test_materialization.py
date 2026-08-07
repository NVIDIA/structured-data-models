import pytest
import torch
from torch import Tensor

from sdm import CategoricalTensor, StringTensor

aten = torch.ops.aten


def _tensor() -> CategoricalTensor:
    return CategoricalTensor(
        code=torch.tensor(
            [
                [[0, 1], [1, 0]],
                [[1, 0], [0, 1]],
            ],
            dtype=torch.int32,
        ),
        categories=(
            StringTensor.from_list(["a", "b"]),
            torch.tensor([10, 20]),
        ),
    )


def _assert_categories(
    actual: tuple[Tensor, ...],
    expected: tuple[Tensor, ...],
) -> None:
    assert len(actual) == len(expected)
    assert all(
        actual_category is expected_category
        for actual_category, expected_category in zip(actual, expected)
    )


def test_index_overloads_preserve_only_unambiguous_category_metadata() -> None:
    tensor = _tensor()

    rows = aten.index_select.default(tensor, 1, torch.tensor([1, 0]))
    assert isinstance(rows, CategoricalTensor)
    assert rows.code.equal(tensor.code.index_select(1, torch.tensor([1, 0])))
    _assert_categories(rows.categories, tensor.categories)

    columns = aten.index_select.default(tensor, -1, torch.tensor([1, 0]))
    assert isinstance(columns, CategoricalTensor)
    _assert_categories(
        columns.categories,
        (tensor.categories[1], tensor.categories[0]),
    )

    advanced_rows = aten.index.Tensor(tensor, (torch.tensor([1, 0]),))
    assert isinstance(advanced_rows, CategoricalTensor)
    _assert_categories(advanced_rows.categories, tensor.categories)

    advanced_columns = aten.index.Tensor(
        tensor,
        (None, None, torch.tensor([1, 0])),
    )
    assert isinstance(advanced_columns, CategoricalTensor)
    _assert_categories(
        advanced_columns.categories,
        (tensor.categories[1], tensor.categories[0]),
    )

    ambiguous = aten.index.Tensor(
        tensor,
        (torch.tensor([1, 0]), None, torch.tensor([1, 0])),
    )
    assert type(ambiguous) is Tensor


def test_cat_and_stack_preserve_only_one_final_column_axis() -> None:
    tensor = _tensor()

    rows = aten.cat.default((tensor, tensor), 0)
    assert isinstance(rows, CategoricalTensor)
    _assert_categories(rows.categories, tensor.categories)

    columns = aten.cat.default((tensor, tensor), -1)
    assert isinstance(columns, CategoricalTensor)
    _assert_categories(columns.categories, tensor.categories * 2)

    stacked = aten.stack.default((tensor, tensor), 1)
    assert isinstance(stacked, CategoricalTensor)
    _assert_categories(stacked.categories, tensor.categories)

    mixed = aten.cat.default((tensor, tensor.code), 0)
    assert type(mixed) is Tensor
    assert mixed.equal(torch.cat((tensor.code, tensor.code), dim=0))

    final_axis = aten.stack.default((tensor, tensor), -1)
    assert type(final_axis) is Tensor
    assert final_axis.equal(torch.stack((tensor.code, tensor.code), dim=-1))


def test_predicate_and_equality_overloads_materialize_plain_results() -> None:
    tensor = _tensor()
    expected_finite = tensor.code >= 0

    assert aten.isnan.default(tensor).equal(~expected_finite)
    assert aten.isfinite.default(tensor).equal(expected_finite)
    assert aten.equal.default(tensor, tensor)
    assert aten.allclose.default(tensor, tensor)
    assert not aten.equal.default(tensor, tensor.code)

    changed_categories = CategoricalTensor(
        code=tensor.code.clone(),
        categories=(
            StringTensor.from_list(["x", "y"]),
            tensor.categories[1].clone(),
        ),
    )
    assert not aten.equal.default(tensor, changed_categories)
    assert not aten.allclose.default(tensor, changed_categories)


def test_unhandled_out_of_place_ops_intentionally_materialize_codes() -> None:
    tensor = _tensor()

    outputs = (
        (tensor + 1, tensor.code + 1),
        (tensor == 1, tensor.code == 1),
        (
            tensor.where(tensor.isfinite(), 0),
            tensor.code.where(tensor.code >= 0, 0),
        ),
        (tensor.repeat(2, 1, 1), tensor.code.repeat(2, 1, 1)),
    )
    for out, expected in outputs:
        assert type(out) is Tensor
        assert out.equal(expected)


def test_unhandled_mutation_is_rejected_without_changing_codes() -> None:
    tensor = _tensor()
    expected = tensor.code.clone()
    version = tensor._version

    with pytest.raises(NotImplementedError, match="cannot mutate"):
        tensor.add_(1)
    with pytest.raises(NotImplementedError, match="cannot mutate"):
        tensor.copy_(tensor)
    with pytest.raises(NotImplementedError, match="cannot mutate"):
        torch.add(tensor, 1, out=tensor)
    with pytest.raises(NotImplementedError, match="cannot mutate"):
        aten.add.out(self=tensor, other=1, out=tensor)
    with pytest.raises(NotImplementedError, match="cannot mutate"):
        aten.add.out(tensor.code, 1, out=tensor)
    with pytest.raises(NotImplementedError, match="cannot mutate"):
        aten.copy_.default(self=tensor, src=tensor)
    with pytest.raises(NotImplementedError, match="cannot mutate"):
        aten._foreach_add_.Scalar([tensor], 1)

    assert tensor.code.equal(expected)
    assert tensor._version == version


def test_plain_out_tensor_is_a_supported_materialization_boundary() -> None:
    tensor = _tensor()
    out = torch.empty_like(tensor.code)

    returned = torch.add(tensor, 1, out=out)

    assert returned is out
    assert out.equal(tensor.code + 1)


def test_mixed_string_subclass_comparison_redispatches_safely() -> None:
    tensor = _tensor()
    strings = StringTensor.from_list(
        [
            [["a", "10"], ["b", "20"]],
            [["b", "10"], ["a", "20"]],
        ]
    )

    left = tensor == strings
    right = strings == tensor

    assert type(left) is Tensor
    assert type(right) is Tensor
    assert not left.any()
    assert not right.any()
