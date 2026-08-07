from typing import cast

import torch
from torch import Tensor

from sdm import StringTensor

aten = torch.ops.aten


def test_tensor_comparison_overloads_broadcast_and_materialize_masks() -> None:
    left = StringTensor.from_list([["a", "b"]])
    right = StringTensor.from_list([["a"], ["b"]])
    expected = torch.tensor([[True, False], [False, True]])

    equal = aten.eq.Tensor(left, right)
    not_equal = aten.ne.Tensor(left, right)

    assert type(equal) is torch.Tensor
    assert equal.equal(expected)
    assert type(not_equal) is torch.Tensor
    assert not_equal.equal(~expected)


def test_string_operator_forms_include_reversed_operands_and_nulls() -> None:
    tensor = StringTensor.from_list(["a", None, "b"])
    expected = torch.tensor([True, False, False])

    assert (tensor == "a").equal(expected)
    assert cast(Tensor, "a" == tensor).equal(expected)  # noqa: SIM300
    assert (tensor != "a").equal(torch.tensor([False, False, True]))
    assert cast(Tensor, "a" != tensor).equal(  # noqa: SIM300
        torch.tensor([False, False, True])
    )


def test_scalar_overloads_treat_non_string_scalars_as_distinct_values() -> (
    None
):
    tensor = StringTensor.from_list(["1", None, "True"])

    for other in (1, 1.0, 1 + 0j, True):
        equal = aten.eq.Scalar(tensor, other)
        not_equal = aten.ne.Scalar(tensor, other)

        assert not equal.any()
        assert not_equal.all()
        assert (tensor == other).equal(equal)
        assert (tensor != other).equal(not_equal)


def test_non_string_tensor_comparison_obeys_broadcasting() -> None:
    tensor = StringTensor.from_list([["a", None]])
    other = torch.tensor([[1], [2]])

    equal = aten.eq.Tensor(tensor, other)
    not_equal = aten.ne.Tensor(tensor, other)

    assert equal.size() == (2, 2)
    assert not equal.any()
    assert not_equal.size() == (2, 2)
    assert not_equal.all()


def test_comparison_out_overloads_mutate_and_return_plain_destination() -> (
    None
):
    tensor = StringTensor.from_list(["a", "b", None])
    other = StringTensor.from_list(["a", "x", "a"])
    out = torch.empty(3, dtype=torch.bool)

    returned = aten.eq.Tensor_out(tensor, other, out=out)
    assert returned is out
    assert out.equal(torch.tensor([True, False, False]))

    returned = aten.ne.Tensor_out(tensor, other, out=out)
    assert returned is out
    assert out.equal(torch.tensor([False, True, False]))

    returned = aten.eq.Scalar_out(tensor, 1, out=out)
    assert returned is out
    assert not out.any()

    returned = aten.ne.Scalar_out(tensor, 1, out=out)
    assert returned is out
    assert out.all()


def test_comparison_out_overloads_resize_empty_destinations() -> None:
    tensor = StringTensor.from_list(["a", "b", None])
    other = StringTensor.from_list(["a", "x", "a"])

    for op, args, expected in (
        (aten.eq.Tensor_out, (tensor, other), [True, False, False]),
        (aten.ne.Tensor_out, (tensor, other), [False, True, False]),
        (aten.eq.Scalar_out, (tensor, 1), [False, False, False]),
        (aten.ne.Scalar_out, (tensor, 1), [True, True, True]),
    ):
        out = torch.empty(0, dtype=torch.bool)

        returned = op(*args, out=out)

        assert returned is out
        assert out.equal(torch.tensor(expected))


def test_none_uses_the_standard_tensor_object_boundary() -> None:
    tensor = StringTensor.from_list(["a", None])

    assert (tensor == None) is False  # noqa: E711
    assert (tensor != None) is True  # noqa: E711
