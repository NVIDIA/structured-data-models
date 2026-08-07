from collections.abc import Callable
from typing import Any

import pytest
import torch

from sdm import VarLenTensor

aten = torch.ops.aten


_INFERENCE_VIEW_OPS: tuple[tuple[str, Callable[[VarLenTensor], Any]], ...] = (
    ("alias.default", lambda tensor: aten.alias.default(tensor)),
    ("detach.default", lambda tensor: aten.detach.default(tensor)),
    ("view.default", lambda tensor: aten.view.default(tensor, [4])),
    (
        "_unsafe_view.default",
        lambda tensor: aten._unsafe_view.default(tensor, [4]),
    ),
    (
        "as_strided.default",
        lambda tensor: aten.as_strided.default(tensor, [2, 1], [2, 1]),
    ),
    ("reshape.default", lambda tensor: aten.reshape.default(tensor, [4])),
    (
        "flatten.using_ints",
        lambda tensor: aten.flatten.using_ints(tensor, 0, -1),
    ),
    ("squeeze.default", lambda tensor: aten.squeeze.default(tensor)),
    ("squeeze.dim", lambda tensor: aten.squeeze.dim(tensor, 0)),
    ("squeeze.dims", lambda tensor: aten.squeeze.dims(tensor, [0])),
    ("unsqueeze.default", lambda tensor: aten.unsqueeze.default(tensor, 0)),
    (
        "expand.default",
        lambda tensor: aten.expand.default(tensor, [2, 2], implicit=True),
    ),
    ("t.default", lambda tensor: aten.t.default(tensor)),
    (
        "transpose.int",
        lambda tensor: aten.transpose.int(tensor, 0, 1),
    ),
    (
        "permute.default",
        lambda tensor: aten.permute.default(tensor, [1, 0]),
    ),
    ("select.int", lambda tensor: aten.select.int(tensor, 0, 0)),
    (
        "slice.Tensor",
        lambda tensor: aten.slice.Tensor(tensor, 1, 0, 2, 1),
    ),
    (
        "narrow.default",
        lambda tensor: aten.narrow.default(tensor, 1, 0, 1),
    ),
    ("unbind.int", lambda tensor: aten.unbind.int(tensor, 0)),
    ("split.Tensor", lambda tensor: aten.split.Tensor(tensor, 1, 0)),
    (
        "split.sizes",
        lambda tensor: aten.split.sizes(tensor, [1, 1], 0),
    ),
    (
        "split.default",
        lambda tensor: aten.split.default(tensor, [1, 1], 0),
    ),
    (
        "split_with_sizes.default",
        lambda tensor: aten.split_with_sizes.default(tensor, [1, 1], 0),
    ),
)


@pytest.mark.parametrize(
    ("name", "operation"),
    [
        pytest.param(name, operation, id=name)
        for name, operation in _INFERENCE_VIEW_OPS
    ],
)
@pytest.mark.parametrize("input_inference", [False, True])
def test_views_preserve_input_inference_state(
    name: str,
    operation: Callable[[VarLenTensor], Any],
    input_inference: bool,
) -> None:
    del name
    with torch.inference_mode(input_inference):
        tensor = VarLenTensor.from_tensor(torch.arange(4).view(2, 2))

    with torch.inference_mode(not input_inference):
        result = operation(tensor)

    outputs = result if isinstance(result, list | tuple) else (result,)
    assert outputs
    assert all(isinstance(out, VarLenTensor) for out in outputs)
    assert all(torch.is_inference(out) is input_inference for out in outputs)
