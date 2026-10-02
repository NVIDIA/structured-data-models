# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inference compilation without compiling memory chunk schedulers."""

import inspect
from typing import Any, cast

import torch
from torch import Tensor

from sdm.cache import KVCacheEntry
from sdm.models.kumo.tabular.cell_embedding import CellEmbedding
from sdm.nn import TransformerBlock


@torch.library.custom_op("sdm::kumo_gelu", mutates_args=())
def _gelu(x: Tensor, approximate: str) -> Tensor:
    # Inductor's fused GELU can amplify rounding differences across layers.
    return torch.nn.functional.gelu(x, approximate=approximate)


@_gelu.register_fake
def _gelu_fake(x: Tensor, approximate: str) -> Tensor:
    return torch.nn.functional.gelu(x, approximate=approximate)


class _GELU(torch.nn.GELU):
    def forward(self, input: Tensor) -> Tensor:  # noqa: A002
        if torch.compiler.is_compiling() and not torch.is_grad_enabled():
            return _gelu(input, self.approximate)
        return super().forward(input)


def _compile_block(block: TransformerBlock, **kwargs: Any) -> None:
    forward = type(block)._forward.__get__(block)
    compiled = torch.compile(forward, **kwargs)

    def run(
        query: Tensor,
        key_value: Tensor | KVCacheEntry | None = None,
        seqused_key_value: Tensor | None = None,
        attn_mask: Tensor | None = None,
        *,
        return_key_value: bool = False,
        out: Tensor | None = None,
    ) -> Tensor | tuple[Tensor, KVCacheEntry]:
        # AOTAutograd cannot merge some symbolic views when out aliases query.
        # Keep the buffer copy outside the compiled computation.
        if out is not None and out.dtype != query.dtype:
            return forward(
                query,
                key_value,
                seqused_key_value,
                attn_mask,
                return_key_value=return_key_value,
                out=out,
            )
        result = compiled(
            query,
            key_value,
            seqused_key_value,
            attn_mask,
            return_key_value=return_key_value,
        )
        if out is None:
            return result
        if return_key_value:
            output, cache = cast(tuple[Tensor, KVCacheEntry], result)
            out.copy_(output)
            return out, cache
        out.copy_(cast(Tensor, result))
        return out

    cast(Any, block)._forward = run


def compile_inference(model: torch.nn.Module, **kwargs: Any) -> None:
    """Compile tensor regions while keeping batching and caches eager."""
    if "isolate_recompiles" not in inspect.signature(torch.compile).parameters:
        raise RuntimeError(
            "KumoTabular.compile requires PyTorch with isolate_recompiles "
            "support (validated on PyTorch 2.14). Eager inference supports "
            "older versions."
        )
    kwargs = {"isolate_recompiles": True, **kwargs}
    backend = kwargs.get("backend", "inductor")
    if backend is None or backend == "inductor":
        if "emulate_precision_casts" not in torch._inductor.list_options():
            raise RuntimeError(
                "KumoTabular.compile requires Inductor's "
                "emulate_precision_casts option to preserve precision."
            )
        options = (
            torch._inductor.list_mode_options(
                kwargs.get("mode"), kwargs.get("dynamic")
            )
            if kwargs.get("mode") is not None
            else {}
        )
        options.update(kwargs.get("options") or {})
        options.setdefault("emulate_precision_casts", True)
        kwargs = {**kwargs, "options": options}
        kwargs.pop("mode", None)

    for module in model.modules():
        for name, child in module.named_children():
            if type(child) is torch.nn.GELU:
                module.add_module(
                    name,
                    _GELU(approximate=child.approximate).train(child.training),
                )

        # Compile the computation, not forward's memory limit and chunks.
        if isinstance(module, CellEmbedding):
            cast(Any, module)._forward = torch.compile(
                type(module)._forward.__get__(module), **kwargs
            )
        elif isinstance(module, TransformerBlock):
            _compile_block(module, **kwargs)

    # The final head and row projection have no chunk or cache management.
    projection = model.get_submodule("row_project")
    if not isinstance(projection, torch.nn.Identity):
        projection.compile(**kwargs)
    model.get_submodule("icl_block.head").compile(**kwargs)
