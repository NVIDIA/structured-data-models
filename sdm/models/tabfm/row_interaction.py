# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modified for the structured-data-models package.

from typing import Any

import torch
from torch import Tensor

from sdm.models.tabfm.attention import Encoder, RMSNorm


class RowInteraction(torch.nn.Module):
    """Combine feature tokens independently within each table row.

    The module folds table and row dimensions together, applies RoPE-enabled
    attention over feature tokens, and returns either the complete token
    sequence or only the flattened leading CLS tokens. When active feature
    counts are supplied, attention keys after ``num_cls + d`` are masked.

    Args:
        channels: Number of channels in every feature token.
        num_blocks: Number of row-attention blocks.
        num_heads: Number of attention heads.
        feedforward_channels: Hidden width of each feed-forward network.
        num_cls: Number of leading CLS/readout tokens.
        rope_theta: Rotary embedding base used across feature positions.
        output_full: Whether to return all tokens. When ``False``, return only
            the flattened leading ``num_cls`` token representations.
        activation: Feed-forward activation used by every attention block.
        device: Device on which to create parameters and buffers.
        dtype: Dtype of parameters and buffers.
    """

    def __init__(
        self,
        channels: int,
        num_blocks: int,
        num_heads: int,
        feedforward_channels: int,
        num_cls: int,
        rope_theta: float = 100_000.0,
        output_full: bool = True,
        activation: str = "swiglu",
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if num_cls <= 0:
            raise ValueError("num_cls must be positive")

        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}
        self.tf_row = Encoder(
            num_blocks=num_blocks,
            channels=channels,
            num_heads=num_heads,
            feedforward_channels=feedforward_channels,
            activation=activation,
            rope_theta=rope_theta,
            **factory_kwargs,
        )
        self.out_ln = RMSNorm(channels, **factory_kwargs)
        self.num_cls = num_cls
        self.output_full = output_full
        self.row_chunk_size: int | None = None

    def _stage(
        self,
        input: Tensor,
        attn_mask: Tensor | None = None,
    ) -> Tensor:
        output = self.tf_row(input, attn_mask=attn_mask)
        if not self.output_full:
            output = output[:, : self.num_cls, :]
        return self.out_ln(output)

    def forward(
        self,
        input: Tensor,
        d: Tensor | None = None,
    ) -> Tensor:
        """Apply within-row feature interaction.

        Args:
            input: Feature and CLS tokens with shape ``[B, T, H, D]``. ``H``
                includes the leading ``num_cls`` tokens.
            d: Optional active feature counts, excluding CLS tokens, with shape
                ``[B]``.

        Returns:
            Tensor with shape ``[B, T, H, D]`` when ``output_full`` is
            ``True``. Otherwise, tensor with shape
            ``[B, T, num_cls * D]``.
        """
        if input.dim() != 4:
            raise ValueError("input must have shape [B, T, H, D]")
        batch_size, num_rows, num_tokens, channels = input.shape
        if num_rows == 0:
            raise ValueError("input must contain at least one row")
        if num_tokens < self.num_cls:
            raise ValueError("input must contain all configured CLS tokens")
        if d is not None:
            if d.shape != (batch_size,):
                raise ValueError("d must have shape [B]")
            if d.is_floating_point():
                raise ValueError("d must have an integer dtype")
            if d.device != input.device:
                raise ValueError("d and input must use the same device")

        # [B, T, H, D] -> [B * T, H, D].
        source = input.reshape(
            batch_size * num_rows,
            num_tokens,
            channels,
        )
        attn_mask = None
        if d is not None:
            active_tokens = d.to(torch.long) + self.num_cls
            token_index = torch.arange(num_tokens, device=input.device)
            valid = token_index[None, :] < active_tokens[:, None]
            attn_mask = valid.repeat_interleave(num_rows, dim=0)
            attn_mask = attn_mask[:, None, None, :]  # [B * T, 1, 1, H].

        chunk_size = self.row_chunk_size
        if chunk_size is None or source.size(0) <= chunk_size:
            output = self._stage(source, attn_mask)
        else:
            if chunk_size <= 0:
                raise ValueError("row_chunk_size must be positive or None")
            # Rows are independent in this stage, so chunking bounds activation
            # memory without changing the context of any output row.
            output = torch.cat(
                [
                    self._stage(
                        source[start : start + chunk_size],
                        None
                        if attn_mask is None
                        else attn_mask[start : start + chunk_size],
                    )
                    for start in range(0, source.size(0), chunk_size)
                ],
                dim=0,
            )

        if self.output_full:
            return output.reshape(
                batch_size,
                num_rows,
                num_tokens,
                channels,
            )
        return output.reshape(batch_size, num_rows, self.num_cls * channels)
