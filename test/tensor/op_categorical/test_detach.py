import pytest
import torch
from torch import Tensor

from sdm import CategoricalTensor, StringTensor

aten = torch.ops.aten


def _is_alias_of(left: Tensor, right: Tensor) -> bool:
    return torch._C._is_alias_of(  # ty: ignore[unresolved-attribute]
        left,
        right,
    )


def _tensor(*, inference: bool = False) -> CategoricalTensor:
    with torch.inference_mode(inference):
        return CategoricalTensor(
            code=torch.tensor(
                [[0, 1], [1, 0]],
                dtype=torch.int32,
            ),
            categories=(
                StringTensor.from_list(["a", "b"]),
                torch.tensor([10.0, 20.0], requires_grad=not inference),
            ),
        )


@pytest.mark.parametrize("create_inference", [False, True])
@pytest.mark.parametrize("run_inference", [False, True])
def test_detach_preserves_type_aliases_and_inference_state(
    create_inference: bool,
    run_inference: bool,
) -> None:
    tensor = _tensor(inference=create_inference)

    with torch.inference_mode(run_inference):
        out = aten.detach.default(tensor)

    assert isinstance(out, CategoricalTensor)
    assert out.is_inference() == tensor.is_inference()
    assert out.size() == tensor.size()
    assert out.stride() == tensor.stride()
    assert out.tolist() == tensor.tolist()
    assert _is_alias_of(out, tensor)
    assert _is_alias_of(out.code, tensor.code)
    assert all(
        _is_alias_of(actual, expected)
        for actual, expected in zip(out.categories, tensor.categories)
    )
    assert not any(category.requires_grad for category in out.categories)
