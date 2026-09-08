# ruff: noqa: D101, D102
import math
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
        *,
        batch_size_limit: int | None = None,
        out: Tensor | None = None,
    ) -> Tensor:  # [..., R, C, D]
        if out is not None and torch.is_grad_enabled():
            raise RuntimeError(
                "'out' is only supported when gradients are disabled"
            )

        *B, R, C = x.size()

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

        if batch_size_limit is not None:
            rows_per_chunk = max(1, batch_size_limit // (math.prod(B) * C))
            xs = x.split(rows_per_chunk, dim=-2)
        else:
            xs = [x]

        start = 0
        for x in xs:
            x = x[..., index].unsqueeze(-1)  # [..., R, C, G, 1]
            angle = x.to(torch.float32) * freq  # [..., R, C, G, F]
            fourier = torch.cat([angle.sin(), angle.cos()], dim=-1)
            fourier = fourier.to(self.num_lin.weight.dtype)

            # Project Fourier features per semantic type:
            x = torch.where(
                grouped_mask,  # [..., 1, C, G, 1]
                self.cat_lin(fourier),  # [..., R, C, G, D]
                self.num_lin(fourier),  # [..., R, C, G, D]
            )

            if out is None:
                out = x.new_empty(*B, R, C, x.size(-1))

            out_chunk = out[..., start : start + x.size(-4), :, :]
            if torch.is_grad_enabled():
                out_chunk[...] = x.sum(dim=-2)
            else:
                torch.sum(input=x, dim=-2, out=out_chunk)
            start += x.size(-4)

        assert out is not None
        return out
