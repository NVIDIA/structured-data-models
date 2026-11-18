# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
from collections import defaultdict
from collections.abc import Callable, Iterable
from typing import Any, Literal, overload

import torch
from torch import Tensor

from sdm.models.kumo.tabular.model import _KumoTabular
from sdm.nn import LogScale

if not hasattr(torch.optim, "Muon"):
    raise RuntimeError("BatchedMuon requires PyTorch >= 2.9.")


class BatchedMuon(torch.optim.Muon):
    """Muon optimizer that batches updates for equal-shaped parameters."""

    def __init__(
        self,
        params: Iterable[Tensor] | Iterable[dict[str, Any]],
        lr: float,
        weight_decay: float,
        adjust_lr_fn: Literal["original", "match_rms_adamw"] | None,
    ) -> None:
        super().__init__(
            params,
            lr=lr,
            weight_decay=weight_decay,
            adjust_lr_fn=adjust_lr_fn,
        )

    @overload
    def step(self, closure: None = None) -> None: ...

    @overload
    def step(self, closure: Callable[[], float]) -> float: ...

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        """Perform one optimizer step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            batches: dict[
                tuple[torch.Size, torch.device, torch.dtype],
                list[tuple[Tensor, Tensor, Tensor]],
            ] = defaultdict(list)
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                state = self.state[parameter]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(parameter.grad)
                key = (parameter.shape, parameter.device, parameter.dtype)
                batches[key].append(
                    (parameter, parameter.grad, state["momentum_buffer"])
                )

            for entries in batches.values():
                parameters, gradients, buffers = map(list, zip(*entries))
                torch._foreach_lerp_(buffers, gradients, 1 - group["momentum"])
                updates = (
                    torch._foreach_lerp(gradients, buffers, group["momentum"])
                    if group["nesterov"]
                    else buffers
                )
                x = torch.stack(updates).bfloat16()  # [B, rows, columns]
                transpose = x.size(-2) > x.size(-1)
                if transpose:
                    x = x.transpose(-2, -1)
                x.div_(
                    x.norm(dim=(-2, -1), keepdim=True).clamp(min=group["eps"])
                )
                a, b, c = group["ns_coefficients"]
                for _ in range(group["ns_steps"]):
                    gram = x @ x.transpose(-2, -1)
                    polynomial = torch.baddbmm(
                        gram, gram, gram, beta=b, alpha=c
                    )
                    x = torch.baddbmm(x, polynomial, x, beta=a)
                if transpose:
                    x = x.transpose(-2, -1)

                rows, columns = parameters[0].shape
                if group["adjust_lr_fn"] in (None, "original"):
                    scale = math.sqrt(max(1, rows / columns))
                elif group["adjust_lr_fn"] == "match_rms_adamw":
                    scale = 0.2 * math.sqrt(max(rows, columns))
                else:
                    raise AssertionError(group["adjust_lr_fn"])
                lr = group["lr"]
                torch._foreach_mul_(parameters, 1 - lr * group["weight_decay"])
                torch._foreach_add_(
                    parameters, list(x.unbind()), alpha=-lr * scale
                )
        return loss


def configure_optimizers(
    model: _KumoTabular,
    *,
    lr: float,
    warmup_proportion: float,
    max_steps: int,
    lr_end_factor: float,
) -> tuple[
    list[torch.optim.Optimizer], list[torch.optim.lr_scheduler.LambdaLR]
]:
    """Configure the optimizers and learning-rate schedulers."""
    # Label projections and output heads use AdamW.
    adamw = {
        id(parameter)
        for name, module in model.named_modules()
        if name.endswith((".y_emb", ".y_lin", ".head"))
        for parameter in module.parameters()
    }
    # Log-scale modules use AdamW.
    scales = {
        id(parameter)
        for module in model.modules()
        if isinstance(module, LogScale)
        for parameter in module.parameters()
    }
    # All other Linear weights go to Muon.
    muon_ids = (
        {
            id(module.weight)
            for module in model.modules()
            if isinstance(module, torch.nn.Linear)
        }
        - adamw
        - scales
    )
    muon, decay, no_decay = [], [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if id(parameter) in muon_ids:
            muon.append(parameter)
        elif id(parameter) in scales or (
            parameter.ndim >= 2
            and not name.endswith((".num_freq", ".cat_freq"))
        ):
            decay.append(parameter)
        else:
            no_decay.append(parameter)
    optimizers: list[torch.optim.Optimizer] = [
        BatchedMuon(
            muon,
            lr=lr,
            weight_decay=0.01,
            adjust_lr_fn="match_rms_adamw",
        ),
        torch.optim.AdamW(
            [{"params": decay}, {"params": no_decay, "weight_decay": 0.0}],
            lr=lr,
            fused=next(model.parameters()).is_cuda,
        ),
    ]

    warmup_steps = round(warmup_proportion * max_steps)

    def lr_factor(step: int) -> float:
        """Return the warmup and cosine learning-rate multiplier."""
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = min(
            1.0,
            (step - warmup_steps) / max(1, max_steps - warmup_steps),
        )
        return lr_end_factor + (1 - lr_end_factor) * 0.5 * (
            1 + math.cos(math.pi * progress)
        )

    schedulers = [
        torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_factor)
        for optimizer in optimizers
    ]
    return optimizers, schedulers
