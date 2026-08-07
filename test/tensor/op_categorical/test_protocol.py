import pytest
import torch
from torch import Tensor

from sdm import CategoricalTensor, StringTensor


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


def test_wrapper_metadata_matches_noncontiguous_code() -> None:
    base = torch.arange(24, dtype=torch.int32).remainder(2).view(4, 3, 2)
    code = base[1:, ::2]
    tensor = CategoricalTensor(
        code=code,
        categories=(torch.tensor([10, 20]), torch.tensor([30, 40])),
    )

    assert tensor.size() == code.size()
    assert tensor.stride() == code.stride()
    assert tensor.storage_offset() == code.storage_offset()
    assert tensor.dtype == code.dtype
    assert tensor.layout == code.layout
    assert tensor.device == code.device
    assert not tensor.requires_grad


def test_tensor_flatten_round_trip_uses_named_tensor_attributes() -> None:
    tensor = _tensor()
    names, context = tensor.__tensor_flatten__()
    inner_tensors = {name: getattr(tensor, name) for name in names}

    assert all(isinstance(value, Tensor) for value in inner_tensors.values())

    out = CategoricalTensor.__tensor_unflatten__(
        inner_tensors,
        context,
        tensor.size(),
        tensor.stride(),
    )

    assert type(out) is type(tensor)
    assert out.size() == tensor.size()
    assert out.stride() == tensor.stride()
    assert out.storage_offset() == tensor.storage_offset()
    assert out.dtype == tensor.dtype
    assert out.layout == tensor.layout
    assert out.device == tensor.device
    assert out.code is tensor.code
    assert all(
        actual is expected
        for actual, expected in zip(out.categories, tensor.categories)
    )


@pytest.mark.parametrize("create_inference", [False, True])
@pytest.mark.parametrize("run_inference", [False, True])
def test_tensor_unflatten_preserves_inference_state(
    create_inference: bool,
    run_inference: bool,
) -> None:
    tensor = _tensor(inference=create_inference)
    names, context = tensor.__tensor_flatten__()
    inner_tensors = {name: getattr(tensor, name) for name in names}

    with torch.inference_mode(run_inference):
        out = CategoricalTensor.__tensor_unflatten__(
            inner_tensors,
            context,
            tensor.size(),
            tensor.stride(),
        )

    assert out.is_inference() == tensor.is_inference()
    assert out.code is tensor.code
    assert out.tolist() == tensor.tolist()
