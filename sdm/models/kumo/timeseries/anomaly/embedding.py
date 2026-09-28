# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor


class DiffusionEmbedding(torch.nn.Module):
    """Embed discrete Kumo-Anomaly diffusion steps.

    Args:
        num_steps: Number of diffusion steps in the lookup table.
        channels: Even sinusoidal embedding width, at least four.
        out_channels: Projection width. Defaults to ``channels``.
        device: The device.
        dtype: The parameter and output dtype.
    """

    embedding: Tensor

    def __init__(
        self,
        num_steps: int,
        channels: int = 128,
        out_channels: int | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if channels < 4 or channels % 2:
            raise ValueError("'channels' must be even and at least four")
        out_channels = channels if out_channels is None else out_channels
        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}
        self.projection1 = torch.nn.Linear(
            in_features=channels, out_features=out_channels, **factory_kwargs
        )
        self.projection2 = torch.nn.Linear(
            in_features=out_channels,
            out_features=out_channels,
            **factory_kwargs,
        )
        torch.nn.init.xavier_uniform_(self.projection1.weight, gain=0.5)
        torch.nn.init.xavier_uniform_(self.projection2.weight, gain=0.5)

        # Build once in float32 on CPU, as in Kumo-TS. Computing large phases
        # in reduced precision or with accelerator math changes the lookup.
        steps = torch.arange(num_steps, device="cpu", dtype=torch.float32)
        frequencies = torch.arange(
            channels // 2, device="cpu", dtype=torch.float32
        )
        frequencies = 10.0 ** (frequencies / (channels // 2 - 1) * 4.0)
        angles = steps[:, None] * frequencies[None, :]
        self.register_buffer(
            "embedding",
            torch.cat([angles.sin(), angles.cos()], dim=-1).to(
                device=self.projection1.weight.device,
                dtype=self.projection1.weight.dtype,
            ),
            persistent=False,
        )

    def forward(self, step: Tensor) -> Tensor:
        """Map integer step indices ``[...]`` to embeddings ``[..., C]``."""
        x = F.silu(self.projection1(self.embedding[step]))
        return F.silu(self.projection2(x))
