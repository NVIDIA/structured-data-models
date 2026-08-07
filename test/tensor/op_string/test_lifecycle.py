from typing import cast

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


def _tensor(*, inference: bool = False) -> StringTensor:
    with torch.inference_mode(inference):
        return StringTensor.from_list(
            [["a", None], ["bb", "c"]],
            offset_dtype=torch.int32,
        )


def test_tensor_flatten_round_trip_preserves_string_representation() -> None:
    tensor = cast(StringTensor, _tensor()[:, 1:])
    names, context = tensor.__tensor_flatten__()
    inner_tensors = {name: getattr(tensor, name) for name in names}

    out = StringTensor.__tensor_unflatten__(
        inner_tensors,
        context,
        tensor.size(),
        tensor.stride(),
    )

    assert isinstance(out, StringTensor)
    assert out.tolist() == tensor.tolist()
    assert out.size() == tensor.size()
    assert out.stride() == tensor.stride()
    assert out.storage_offset() == tensor.storage_offset()
    assert out.dtype == torch.uint8
    assert out.device == tensor.device
    assert out._data is tensor._data
    assert out._offset is tensor._offset
    assert out._valid is tensor._valid


@pytest.mark.parametrize("create_inference", [False, True])
@pytest.mark.parametrize("run_inference", [False, True])
def test_detach_preserves_string_type_aliases_and_inference_state(
    create_inference: bool,
    run_inference: bool,
) -> None:
    tensor = _tensor(inference=create_inference)

    with torch.inference_mode(run_inference):
        out = aten.detach.default(tensor)

    assert isinstance(out, StringTensor)
    assert out.is_inference() == tensor.is_inference()
    assert out.tolist() == tensor.tolist()
    assert _is_alias_of(out, tensor)
    assert _is_alias_of(out._data, tensor._data)
    assert out._offset is tensor._offset
    assert out._valid is tensor._valid


@pytest.mark.parametrize("inference", [False, True])
def test_views_preserve_string_type_aliases_and_inference_state(
    inference: bool,
) -> None:
    tensor = _tensor(inference=inference)

    with torch.inference_mode(not inference):
        outputs = (
            tensor.view(4),
            tensor.transpose(0, 1),
            tensor[:, 1:],
            tensor.unsqueeze(0),
        )

    expected = (
        ["a", None, "bb", "c"],
        [["a", "bb"], [None, "c"]],
        [[None], ["c"]],
        [[["a", None], ["bb", "c"]]],
    )
    for out, values in zip(outputs, expected):
        assert isinstance(out, StringTensor)
        assert out.is_inference() == tensor.is_inference()
        assert out.tolist() == values
        assert _is_alias_of(out, tensor)


def test_clone_and_to_copy_preserve_type_with_independent_storage() -> None:
    tensor = _tensor()

    for out in (
        aten.clone.default(tensor),
        aten._to_copy.default(tensor),
    ):
        assert isinstance(out, StringTensor)
        assert out.tolist() == tensor.tolist()
        assert not _is_alias_of(out, tensor)
        assert not _is_alias_of(out._data, tensor._data)
        assert not _is_alias_of(out._offset, tensor._offset)
        assert out._valid is not None
        assert tensor._valid is not None
        assert not _is_alias_of(out._valid, tensor._valid)
