import torch
from torch import Tensor

from sdm import CategoricalTensor, StringTensor

aten = torch.ops.aten


def _is_alias_of(left: Tensor, right: Tensor) -> bool:
    return torch._C._is_alias_of(  # ty: ignore[unresolved-attribute]
        left,
        right,
    )


def _tensor() -> CategoricalTensor:
    return CategoricalTensor(
        code=torch.tensor(
            [[0, 1], [1, 0]],
            dtype=torch.int32,
        ),
        categories=(
            StringTensor.from_list(["a", "b"]),
            torch.tensor([10.0, 20.0], requires_grad=True),
        ),
    )


def test_clone_and_to_copy_have_independent_inner_storage() -> None:
    tensor = _tensor()

    for out in (
        aten.clone.default(tensor),
        aten._to_copy.default(tensor),
    ):
        assert isinstance(out, CategoricalTensor)
        assert out.tolist() == tensor.tolist()
        assert not _is_alias_of(out, tensor)
        assert not _is_alias_of(out.code, tensor.code)
        string, numerical = out.categories
        expected_string, expected_numerical = tensor.categories
        assert isinstance(string, StringTensor)
        assert isinstance(expected_string, StringTensor)
        assert not _is_alias_of(string._data, expected_string._data)
        assert not _is_alias_of(
            string._offset,
            expected_string._offset,
        )
        assert not _is_alias_of(numerical, expected_numerical)


def test_to_overloads_preserve_or_materialize_by_dtype() -> None:
    tensor = _tensor()

    assert aten.to.dtype(tensor, torch.int32) is tensor
    assert (
        aten.to.device(
            tensor,
            torch.device("cpu"),
            torch.int32,
        )
        is tensor
    )

    out = aten.to.dtype_layout(tensor, dtype=torch.int64)
    assert isinstance(out, CategoricalTensor)
    assert out.dtype == torch.int64
    assert out.tolist() == tensor.tolist()

    out = aten.to.other(tensor, torch.empty((), dtype=torch.int64))
    assert isinstance(out, CategoricalTensor)
    assert out.dtype == torch.int64
    assert out.tolist() == tensor.tolist()

    materialized = aten.to.dtype(tensor, torch.float32)
    assert type(materialized) is Tensor
    assert materialized.equal(tensor.code.to(torch.float32))
