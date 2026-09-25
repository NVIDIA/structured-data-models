# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101

from typing import Any, cast

import torch
from torch import Tensor
from torch.nn import GELU, Linear, RMSNorm, Sequential

from sdm import _compiled_regions
from sdm._compiled_regions import rms_norm
from torch.nn.modules import module as module_hooks
from torch.overrides import _get_current_function_mode
from torch.utils._python_dispatch import _get_current_dispatch_mode
from sdm.nn import QueryScaling, RotaryEmbedding, TransformerBlock


def _has_hooks(module: torch.nn.Module) -> bool:
    return bool(
        module._forward_pre_hooks
        or module._forward_hooks
        or module._backward_pre_hooks
        or module._backward_hooks
        or module_hooks._global_forward_pre_hooks
        or module_hooks._global_forward_hooks
        or module_hooks._global_backward_pre_hooks
        or module_hooks._global_backward_hooks
    )


class _RMSNorm(RMSNorm):
    def forward(self, x: Tensor, rope: RotaryEmbedding | None = None) -> Tensor:
        eligible = (
            _compiled_regions.is_enabled()
            and not torch.is_grad_enabled()
            and not torch.compiler.is_compiling()
            and x.is_cuda
            and torch.is_autocast_enabled("cuda")
            and torch.get_autocast_dtype("cuda") == torch.float16
            and x.dtype == torch.float16
            and type(x) is Tensor
            and _get_current_function_mode() is None
            and _get_current_dispatch_mode() is None
        )
        if eligible and rope is None:
            eligible = (
                self.normalized_shape == (128,)
                and type(self.weight) is torch.nn.Parameter
                and self.weight.shape == (128,)
                and self.weight.is_contiguous()
                and self.weight.requires_grad
                and self.weight.dtype == torch.float32
                and self.weight.device == x.device
                and self.weight.data_ptr() % 16 == 0
                and self.eps is None
            )
        elif eligible:
            eligible = (
                self.normalized_shape == (32,)
                and self.weight is None
                and self.eps == 1e-6
                and type(rope) is RotaryEmbedding
                and x.ndim >= 4
                and x.shape[-3] >= 4
                and x.shape[-2:] == (4, 32)
                and rope.channels == 32
                and rope.rotary_channels == 32
                and rope.layout == "split_half"
                and type(rope.inv_freq) is torch.nn.Parameter
                and rope.inv_freq.shape == (16,)
                and rope.inv_freq.is_contiguous()
                and not rope.inv_freq.requires_grad
                and rope.inv_freq.dtype == torch.float32
                and rope.inv_freq.device == x.device
            )
        if not eligible or _has_hooks(self) or (rope is not None and _has_hooks(rope)):
            return super().forward(x if rope is None else rope(x))
        eps = torch.finfo(torch.float32).eps if self.eps is None else self.eps
        return rms_norm(
            x,
            weight=self.weight,
            eps=eps,
            dtype=torch.float32 if self.weight is None else torch.float16,
            rope=rope,
        )


class _RoPERMSNorm(Sequential):
    def forward(self, input: Tensor) -> Tensor:
        rope, norm = self
        if not _compiled_regions.is_enabled() or any(_has_hooks(m) for m in (self, rope, norm)):
            return super().forward(input)
        return norm(input, rope=rope)


class KumoTabularTransformerBlock(TransformerBlock):
    def __init__(
        self,
        channels: int,
        num_heads: int,
        query_scaling: QueryScaling | None,
        rope: RotaryEmbedding | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}

        query_transforms: list[torch.nn.Module] = []
        key_transforms: list[torch.nn.Module] = []
        if rope is not None:
            query_transforms.append(rope)
            key_transforms.append(rope)
        query_transforms.append(
            _RMSNorm(
                channels // num_heads,
                eps=1e-6,
                elementwise_affine=False,
                **factory_kwargs,
            )
        )
        key_transforms.append(
            _RMSNorm(
                channels // num_heads,
                eps=1e-6,
                elementwise_affine=False,
                **factory_kwargs,
            )
        )

        mlp = Sequential(
            _RMSNorm(channels, **factory_kwargs),
            Linear(channels, 2 * channels, **factory_kwargs),
            GELU(),
            Linear(2 * channels, channels, **factory_kwargs),
        )
        torch.nn.init.zeros_(cast(Linear, mlp[-1]).weight)
        torch.nn.init.zeros_(cast(Linear, mlp[-1]).bias)

        transform = Sequential if rope is None else _RoPERMSNorm
        super().__init__(
            channels=channels,
            num_query_heads=num_heads,
            mlp=mlp,
            query_norm=_RMSNorm(channels, **factory_kwargs),
            key_value_norm=_RMSNorm(channels, **factory_kwargs),
            query_transform=transform(*query_transforms),
            key_transform=transform(*key_transforms),
            query_scaling=query_scaling,
            **factory_kwargs,
        )

    def forward(self, *args: Any, **kwargs: Any):
        if _compiled_regions.is_enabled() and (
            _get_current_function_mode() is not None
            or _get_current_dispatch_mode() is not None
            or any(_has_hooks(module) for module in self.modules())
        ):
            with _compiled_regions.native_only():
                return super().forward(*args, **kwargs)
        return super().forward(*args, **kwargs)

    def peak_bytes_per_example(
        self,
        element_size: int,
        query_length: int,
        key_value_length: int | None = None,
    ) -> int:
        r""":meta private:"""  # noqa: D415
        length = max(query_length, key_value_length or 0)
        factor = 15 if element_size <= 2 else 8
        return factor * length * element_size * self.attn.q_dim


if __name__ == "__main__":
    from sdm.nn import LogScale
    from sdm.testing.memory import benchmark_transformer_block_memory_peak

    benchmark_transformer_block_memory_peak(
        block=lambda channels, num_heads: KumoTabularTransformerBlock(
            channels=channels,
            num_heads=num_heads,
            query_scaling=LogScale(num_heads=num_heads),
        ),
        channels_and_heads=[(256, 4), (512, 4)],
    )
