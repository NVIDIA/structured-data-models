# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D103

import torch
import torch.nn.functional as F
from torch import Tensor


def embed(encoded: Tensor) -> Tensor:
    pooled = encoded.float().mean(dim=-3).mean(dim=-2)
    pooled = pooled.nan_to_num(nan=0, posinf=0, neginf=0)
    return F.normalize(pooled, dim=-1, eps=1e-12)


def knn(
    query: Tensor,  # [..., D]
    key: Tensor,  # [..., N, D]
    value: Tensor,  # [..., N, C, H]
    k: int,
    temperature: float | None,
) -> Tensor:  # [..., C, H]
    with torch.amp.autocast(query.device.type, enabled=False):
        sim = query.float().unsqueeze(-2) @ key.float().transpose(-1, -2)
        sim, index = sim.squeeze(-2).topk(k=min(k, key.size(-2)), dim=-1)

    if temperature is None or temperature < 0:
        weight = torch.ones_like(sim) / sim.size(-1)
    else:
        weight = (sim / max(temperature, 1e-12)).softmax(dim=-1)

    neighbor = value.gather(
        dim=-3,
        index=index[..., None, None].expand(*index.size(), *value.size()[-2:]),
    ).float()  # [..., K, C, H]

    valid = neighbor.isfinite()
    weight = weight[..., None, None] * valid
    total = (weight * neighbor.masked_fill(~valid, 0)).sum(dim=-3)
    return total / weight.sum(dim=-3).clamp_min(1e-12)
