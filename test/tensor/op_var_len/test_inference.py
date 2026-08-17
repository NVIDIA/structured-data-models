from collections.abc import Callable

import pytest
import torch
from torch import Tensor

from sdm import VarLenTensor

aten = torch.ops.aten


_MATERIALIZING_OPS: tuple[
    tuple[str, Callable[[VarLenTensor], Tensor]], ...
] = (
    (
        "to.dtype_layout",
        lambda tensor: aten.to.dtype_layout(tensor, dtype=torch.float64),
    ),
    ("to.dtype", lambda tensor: aten.to.dtype(tensor, torch.float64)),
    (
        "to.device",
        lambda tensor: aten.to.device(
            tensor,
            torch.device("cpu"),
            torch.float64,
        ),
    ),
    (
        "to.other",
        lambda tensor: aten.to.other(
            tensor,
            torch.empty((), dtype=torch.float64),
        ),
    ),
    (
        "_to_copy.default",
        lambda tensor: aten._to_copy.default(tensor, dtype=torch.float64),
    ),
    ("clone.default", lambda tensor: aten.clone.default(tensor)),
    (
        "contiguous.default",
        lambda tensor: aten.contiguous.default(tensor.t()),
    ),
    (
        "reshape.copy",
        lambda tensor: aten.reshape.default(tensor.t(), [4]),
    ),
    ("isnan.default", lambda tensor: aten.isnan.default(tensor)),
    ("isfinite", lambda tensor: torch.isfinite(tensor)),
    (
        "masked_select.default",
        lambda tensor: aten.masked_select.default(
            tensor,
            torch.tensor([[True, False], [False, True]]),
        ),
    ),
    (
        "index_select.default",
        lambda tensor: aten.index_select.default(
            tensor,
            0,
            torch.tensor([1, 0]),
        ),
    ),
    (
        "take.default",
        lambda tensor: aten.take.default(tensor, torch.tensor([3, 0])),
    ),
    (
        "index.Tensor",
        lambda tensor: aten.index.Tensor(tensor, [torch.tensor([1, 0])]),
    ),
    ("cat.default", lambda tensor: aten.cat.default([tensor, tensor])),
    ("stack.default", lambda tensor: aten.stack.default([tensor, tensor])),
)


@pytest.mark.parametrize(
    ("name", "operation"),
    [
        pytest.param(name, operation, id=name)
        for name, operation in _MATERIALIZING_OPS
    ],
)
@pytest.mark.parametrize("input_inference", [False, True])
def test_materializing_ops_follow_ambient_inference_mode(
    name: str,
    operation: Callable[[VarLenTensor], Tensor],
    input_inference: bool,
) -> None:
    del name
    with torch.inference_mode(input_inference):
        tensor = VarLenTensor.from_list([[[0.0], None], [[1.0, 2.0], [3.0]]])

    output_inference = not input_inference
    with torch.inference_mode(output_inference):
        out = operation(tensor)

    assert torch.is_inference(out) is output_inference
