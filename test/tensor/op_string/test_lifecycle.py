from typing import cast

import torch

from sdm import StringTensor


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
