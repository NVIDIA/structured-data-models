# ruff: noqa: D101, D102
import math
import os
from typing import Any, Literal

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
        self.channels = channels
        self.group_size = group_size
        self.num_frequencies = num_frequencies

        self.num_freq = torch.nn.Parameter(
            torch.randn(group_size, num_frequencies, **factory_kwargs)
        )
        self.cat_freq = torch.nn.Parameter(
            torch.randn(group_size, num_frequencies, **factory_kwargs)
        )

        self.num_lin = Linear(2 * num_frequencies, channels, **factory_kwargs)
        self.cat_lin = Linear(2 * num_frequencies, channels, **factory_kwargs)

    def _group_index(self, num_columns: int, device: torch.device) -> Tensor:
        index = torch.arange(num_columns, device=device)
        shift = 2 ** torch.arange(self.group_size, device=device) - 1
        return (
            index.view(num_columns, 1) + shift.view(1, self.group_size)
        ) % num_columns

    def forward(
        self,
        x: Tensor,  # [..., R, C],
        categorical_mask: Tensor,  # [..., C],
        *,
        batch_size_limit: int | Literal["auto"] | None = None,
        out: Tensor | None = None,
    ) -> Tensor:  # [..., R, C, D]
        if out is not None and torch.is_grad_enabled():
            raise RuntimeError(
                "'out' is only supported when gradients are disabled"
            )

        *B, R, C = x.size()

        # Feature grouping:
        index = self._group_index(C, x.device)
        x = x[..., index]  # [..., R, C, G]

        # Compute Fourier features per semantic type (row-agnostic):
        categorical_mask = categorical_mask[..., None, index]  # [..., 1, C, G]
        freq = torch.where(
            categorical_mask.unsqueeze(-1),  # [..., 1, C, G, 1]
            self.cat_freq.to(torch.float32),  # [G, F]
            self.num_freq.to(torch.float32),  # [G, F]
        )  # [..., 1, C, G, F]

        # Gather weights and biases per semantic type (row-agnostic):
        dtype = (
            torch.get_autocast_dtype(x.device.type)
            if torch.is_autocast_enabled(x.device.type)
            and not torch.is_grad_enabled()
            else self.num_lin.weight.dtype
        )

        weight = torch.where(
            categorical_mask[..., None, None],  # [..., G, 1, 1]
            self.cat_lin.weight.to(dtype).view(
                *(1,) * categorical_mask.dim(), *self.cat_lin.weight.size()
            ),
            self.num_lin.weight.to(dtype).view(
                *(1,) * categorical_mask.dim(), *self.num_lin.weight.size()
            ),
        )  # [..., 1, C, G, D, 2F]

        bias = torch.where(
            categorical_mask[..., None],
            self.cat_lin.bias.view(*(1,) * categorical_mask.dim(), -1),
            self.num_lin.bias.view(*(1,) * categorical_mask.dim(), -1),
        ).sum(dim=-2)  # [..., 1, C, D]
        bias = bias.to(dtype)

        if torch.is_grad_enabled() or torch.compiler.is_compiling():
            return self._forward(x, freq, weight, bias, out=out)

        if batch_size_limit == "auto":
            batch_size_limit = None
            if x.is_cuda:
                if torch.is_autocast_enabled(x.device.type):
                    element_size = torch.empty(
                        size=(),
                        dtype=torch.get_autocast_dtype(x.device.type),
                    ).element_size()
                else:
                    element_size = x.element_size()

                bytes_per_example = (
                    2 * self.group_size * self.num_frequencies * element_size
                    + 2 * self.channels * element_size
                )

                fixed_bytes = (
                    freq.numel() * freq.element_size()
                    + weight.numel() * weight.element_size()
                    + bias.numel() * bias.element_size()
                )

                memory_limit = int(
                    torch.cuda.get_device_properties(x.device).total_memory
                    * torch.cuda.get_per_process_memory_fraction(x.device)
                    * float(os.getenv("SDM_CHUNK_MEMORY_FRACTION", "0.05"))
                )
                memory_limit -= fixed_bytes
                batch_size_limit = memory_limit // max(bytes_per_example, 1)
                batch_size_limit = max(batch_size_limit, 1)

        if batch_size_limit is None:
            return self._forward(x, freq, weight, bias, out=out)

        if out is None:
            out = x.new_empty(
                (*B, R, C, self.channels),
                dtype=torch.get_autocast_dtype(x.device.type)
                if torch.is_autocast_enabled(x.device.type)
                else x.dtype,
            )

        rows_per_chunk = max(1, batch_size_limit // (math.prod(B) * C))
        for start in range(0, R, rows_per_chunk):
            self._forward(
                x=x[..., start : start + rows_per_chunk, :, :],
                freq=freq,
                weight=weight,
                bias=bias,
                out=out[..., start : start + rows_per_chunk, :, :],
            )

        return out

    def _forward(
        self,
        x: Tensor,  # [..., G]
        freq: Tensor,  # [..., G, F]
        weight: Tensor,  # [..., G, D, 2F]
        bias: Tensor,  # [..., D]
        *,
        out: Tensor | None = None,
    ) -> Tensor:

        x = x.to(torch.float32).unsqueeze(-1) * freq  # [..., G, F]
        fourier = x.new_empty(
            (*x.size()[:-1], 2 * x.size(-1)),
            dtype=weight.dtype,
        )
        if torch.is_grad_enabled():
            fourier[..., : x.size(-1)] = x.sin()
            fourier[..., x.size(-1) :] = x.cos()
        else:
            torch.sin(x, out=fourier[..., : x.size(-1)])
            torch.cos(x, out=fourier[..., x.size(-1) :])
        del x

        if out is None:
            out = torch.einsum("...gf,...gdf->...d", fourier, weight)
        else:
            out.copy_(torch.einsum("...gf,...gdf->...d", fourier, weight))

        out += bias.to(out.dtype)

        return out

    def peak_bytes_per_example(self, element_size: int) -> int:
        r""":meta private:"""  # noqa: D415
        return (5 * self.group_size * self.num_frequencies * 4) + (
            self.channels * element_size
        )
