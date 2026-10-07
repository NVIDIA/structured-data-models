"""Diagnostic only: keep native LayerNorm inside a compiled graph."""
# ruff: noqa: A002, D103

import torch
from torch import Tensor

_original_forward = torch.nn.LayerNorm.forward


@torch.library.custom_op("precision_probe::layer_norm", mutates_args=())
def native_layer_norm(
    input: Tensor,
    normalized_shape: list[int],
    weight: Tensor | None,
    bias: Tensor | None,
    eps: float,
) -> Tensor:
    return torch.nn.functional.layer_norm(
        input, normalized_shape, weight, bias, eps
    )


@native_layer_norm.register_fake
def _fake(
    input: Tensor,
    normalized_shape: list[int],
    weight: Tensor | None,
    bias: Tensor | None,
    eps: float,
) -> Tensor:
    return input.new_empty(input.shape)


# Match CUDA LayerNorm autocast without changing parameter storage dtypes.
torch.library.register_autocast(native_layer_norm, "cuda", torch.float32)


def _forward(self: torch.nn.LayerNorm, input: Tensor) -> Tensor:
    if not torch.compiler.is_compiling() or torch.is_grad_enabled():
        return _original_forward(self, input)
    return native_layer_norm(
        input,
        list(self.normalized_shape),
        self.weight,
        self.bias,
        self.eps,
    )


torch.nn.LayerNorm.forward = _forward
