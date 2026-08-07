from typing import cast

import pytest
import torch
from torch import Tensor

from sdm import VarLenTensor


def _is_alias(left: Tensor, right: Tensor) -> bool:
    return torch._C._is_alias_of(  # ty: ignore[unresolved-attribute]
        left,
        right,
    )


def test_public_metadata_tracks_logical_layout() -> None:
    base = VarLenTensor.from_list(
        [[[0.0], [1.0, 2.0], [3.0]], [[4.0], [5.0], [6.0, 7.0]]]
    )
    tensor = cast(VarLenTensor, base[:, 1:])

    assert tensor.size() == (2, 2)
    assert tensor.stride() == (3, 1)
    assert tensor.storage_offset() == 1
    assert tensor.dtype == tensor._data.dtype == torch.float32
    assert tensor.device == tensor._data.device
    assert tensor.layout == torch.strided
    assert tensor.requires_grad == tensor._data.requires_grad


@pytest.mark.parametrize("nullable", [False, True])
@pytest.mark.parametrize("inference", [False, True])
def test_tensor_flatten_round_trip(
    nullable: bool,
    inference: bool,
) -> None:
    values = [[1], None, [2, 3]] if nullable else [[1], [], [2, 3]]
    with torch.inference_mode(inference):
        tensor = cast(VarLenTensor, VarLenTensor.from_list(values)[1:])
    names, metadata = tensor.__tensor_flatten__()

    with torch.inference_mode(not inference):
        out = VarLenTensor.__tensor_unflatten__(
            {name: getattr(tensor, name) for name in names},
            metadata,
            tensor.size(),
            tensor.stride(),
        )

    assert type(out) is type(tensor)
    assert out.size() == tensor.size()
    assert out.stride() == tensor.stride()
    assert out.storage_offset() == tensor.storage_offset()
    assert out.is_inference() is inference
    assert out.tolist() == tensor.tolist()
    assert out._data is tensor._data
    assert out._offset is tensor._offset
    assert out._valid is tensor._valid


def test_dynamic_compiled_views_preserve_protocol() -> None:
    compiled = torch.compile(
        lambda tensor: tensor.unsqueeze(0).transpose(0, 1),
        backend="eager",
        fullgraph=True,
        dynamic=True,
    )

    for values in ([[1], [2, 3]], [[1], None, [2], [3, 4]]):
        tensor = VarLenTensor.from_list(values)
        out = compiled(tensor)

        assert isinstance(out, VarLenTensor)
        assert out.size() == (len(values), 1)
        assert out.tolist() == [[value] for value in values]
        assert _is_alias(out, tensor)
