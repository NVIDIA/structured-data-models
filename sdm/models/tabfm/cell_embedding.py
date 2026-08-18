# ruff: noqa: D101, D102

from typing import Any

import torch
from torch import Tensor
from torch.nn import Linear


class CellEmbedding(torch.nn.Module):
    def __init__(
        self,
        channels: int,
        group_size: int,
        num_frequencies: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}
        self.group_size = group_size

        self.num_freq = torch.nn.Parameter(
            torch.randn(group_size, num_frequencies, **factory_kwargs)
        )
        self.cat_freq = torch.nn.Parameter(
            torch.randn(group_size, num_frequencies, **factory_kwargs)
        )

        self.num_lin = Linear(2 * num_frequencies, channels, **factory_kwargs)
        self.cat_lin = Linear(2 * num_frequencies, channels, **factory_kwargs)

    def forward(
        self,
        x: Tensor,  # [..., R, C],
        categorical_mask: Tensor,  # [..., C],
    ) -> Tensor:  # [..., R, C, D]
        C = x.size(-1)

        # Feature grouping:
        index = torch.arange(C, device=x.device)
        shift = 2 ** torch.arange(self.group_size, device=x.device) - 1
        index = (index.view(C, 1) + shift.view(1, self.group_size)) % C

        # Compute Fourier features per semantic type:
        grouped_mask = categorical_mask[..., index]  # [..., C, G]
        grouped_mask = grouped_mask.unsqueeze(-1)
        grouped_mask = grouped_mask.unsqueeze(-4)  # [..., 1, C, G, 1]
        freq = torch.where(
            grouped_mask,  # [..., 1, C, G, 1]
            self.cat_freq.to(torch.float32),  # [G, F]
            self.num_freq.to(torch.float32),  # [G, F]
        )  # [..., 1, C, G, F]
        angle = x.unsqueeze(-1).to(torch.float32) * freq  # [..., R, C, G, F]
        fourier = torch.cat([angle.sin(), angle.cos()], dim=-1)
        fourier = fourier.to(self.num_lin.weight.dtype)

        # Project Fourier features per semantic type:
        return torch.where(
            grouped_mask,  # [..., 1, C, G, 1]
            self.cat_lin(fourier),  # [..., R, C, G, D]
            self.num_lin(fourier),  # [..., R, C, G, D]
        ).sum(dim=-2)  # [..., R, C, D]
