# ruff: noqa: D101, D102
import math
from typing import Any

import torch
import torch.nn.functional as F
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
        *,
        batch_size_limit: int | None = 1_700_000,
        num_prefix_columns: int = 0,
        num_categorical_columns: int | None = None,
    ) -> Tensor:  # [..., R, P + C, D]
        *B, R, C = x.size()
        batch_size = max(1, math.prod(B))
        projection_dtype = (
            torch.get_autocast_dtype(x.device.type)
            if torch.is_autocast_enabled(x.device.type)
            else self.num_lin.weight.dtype
        )

        # Feature grouping:
        index = torch.arange(C, device=x.device)
        shift = 2 ** torch.arange(self.group_size, device=x.device) - 1
        index = (index.view(C, 1) + shift.view(1, self.group_size)) % C

        if num_categorical_columns == 0:
            grouped_mask = None
            freq = self.num_freq.to(torch.float32)
            lin = self.num_lin
            weight = lin.weight.repeat(1, self.group_size)
            bias = (
                lin.bias.to(projection_dtype) * self.group_size
                if lin.bias is not None
                else None
            )
        elif num_categorical_columns == C:
            grouped_mask = None
            freq = self.cat_freq.to(torch.float32)
            lin = self.cat_lin
            weight = lin.weight.repeat(1, self.group_size)
            bias = (
                lin.bias.to(projection_dtype) * self.group_size
                if lin.bias is not None
                else None
            )
        else:
            # Compute Fourier features per semantic type:
            grouped_mask = categorical_mask[..., index]  # [..., C, G]
            grouped_mask = grouped_mask.unsqueeze(-1)
            grouped_mask = grouped_mask.unsqueeze(-4)  # [..., 1, C, G, 1]
            freq = torch.where(
                grouped_mask,  # [..., 1, C, G, 1]
                self.cat_freq.to(torch.float32),  # [G, F]
                self.num_freq.to(torch.float32),  # [G, F]
            )  # [..., 1, C, G, F]
            lin = None
            mask = grouped_mask.squeeze(-4)  # [..., C, G, 1]
            weight = torch.where(
                mask.unsqueeze(-1),  # [..., C, G, 1, 1]
                self.cat_lin.weight,  # [D, 2 * F]
                self.num_lin.weight,  # [D, 2 * F]
            )  # [..., C, G, D, 2 * F]
            weight = weight.to(projection_dtype)
            bias = torch.where(
                mask,  # [..., C, G, 1]
                self.cat_lin.bias,  # [D]
                self.num_lin.bias,  # [D]
            )  # [..., C, G, D]
            bias = bias.to(projection_dtype)

        if batch_size_limit is not None:
            rows_per_chunk = max(1, batch_size_limit // (batch_size * C))
            xs = x.split(rows_per_chunk, dim=-2)
        else:
            xs = [x]

        start = 0
        out: Tensor | None = None
        for i, x in enumerate(xs):
            x = x[..., index].unsqueeze(-1)  # [..., R, C, G, 1]
            angle = x.to(torch.float32) * freq  # [..., R, C, G, F]
            fourier = torch.cat([angle.sin(), angle.cos()], dim=-1)
            fourier = fourier.to(projection_dtype)

            # Project Fourier features per semantic type:
            if lin is not None:
                x = F.linear(fourier.flatten(-2, -1), weight, bias)
            else:
                x = torch.einsum('...rcgf,...cgdf->...rcgd', fourier, weight)
                x = x + bias.unsqueeze(-4)
                x = x.sum(dim=-2)
            x = x.to(projection_dtype)  # [..., R, C, D]

            if len(xs) == 1 and num_prefix_columns == 0:
                out = x
                continue

            if i == 0:
                out = x.new_empty(*B, R, num_prefix_columns + C, x.size(-1))
            assert out is not None
            out[
                ...,
                start : start + x.size(-3),
                num_prefix_columns:,
                :,
            ] = x
            start += x.size(-3)

        assert out is not None
        return out
