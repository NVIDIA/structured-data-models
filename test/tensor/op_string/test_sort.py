import pytest
import torch
from torch import Tensor

from sdm import StringTensor

aten = torch.ops.aten


def _is_alias_of(left: Tensor, right: Tensor) -> bool:
    return torch._C._is_alias_of(  # ty: ignore[unresolved-attribute]
        left,
        right,
    )


def test_sort_default_and_stable_overloads_return_values_and_indices() -> None:
    tensor = StringTensor.from_list(["b", "a", "b", None, "a"])
    expected_values = ["a", "a", "b", "b", None]
    expected_indices = torch.tensor([1, 4, 0, 2, 3])

    default_values, default_indices = aten.sort.default(tensor)
    stable_values, stable_indices = aten.sort.stable(
        tensor,
        stable=True,
    )

    for values, indices in (
        (default_values, default_indices),
        (stable_values, stable_indices),
    ):
        assert isinstance(values, StringTensor)
        assert values.tolist() == expected_values
        assert indices.equal(expected_indices)


@pytest.mark.parametrize("dim", [-1, 0])
@pytest.mark.parametrize("stable", [None, False, True])
@pytest.mark.parametrize("value", ["é", None])
def test_sort_scalar_matches_tensor_dimension_and_copy_semantics(
    dim: int,
    stable: bool | None,
    value: str | None,
) -> None:
    tensor = StringTensor.from_list(value)

    if stable is None:
        values, indices = aten.sort.default(tensor, dim)
    else:
        values, indices = aten.sort.stable(
            tensor,
            stable=stable,
            dim=dim,
        )

    assert isinstance(values, StringTensor)
    assert values.tolist() == value
    assert indices.equal(torch.zeros((), dtype=torch.int64))
    assert not _is_alias_of(values, tensor)


def test_argsort_uses_string_sort_indices() -> None:
    tensor = StringTensor.from_list(["bb", "a", "", "é"])

    indices = torch.argsort(tensor, descending=True)

    assert indices.equal(torch.tensor([3, 0, 1, 2]))


def test_sort_rejects_unsupported_dimensions_and_multidimensional_input() -> (
    None
):
    scalar = StringTensor.from_list("a")
    with pytest.raises(IndexError, match="Dimension out of range"):
        aten.sort.default(scalar, 1)

    matrix = StringTensor.from_list([["a", "b"], ["c", "d"]])
    with pytest.raises(NotImplementedError, match="one-dimensional"):
        aten.sort.default(matrix, 0)


def test_sort_out_overloads_are_explicitly_unsupported() -> None:
    tensor = StringTensor.from_list(["b", "a"])
    values = StringTensor.from_list(["", ""])
    indices = torch.empty(2, dtype=torch.int64)

    with pytest.raises(NotImplementedError, match=r"sort\.values"):
        aten.sort.values(
            tensor,
            values=values,
            indices=indices,
        )
    with pytest.raises(NotImplementedError, match=r"sort\.values_stable"):
        aten.sort.values_stable(
            tensor,
            stable=True,
            values=values,
            indices=indices,
        )
