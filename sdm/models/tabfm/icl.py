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

from typing import Any, cast

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear

from sdm.cache import Cache
from sdm.models.tabfm.attention import Encoder, RMSNorm
from sdm.models.tabfm.mlp import MLP


class OneHotAndLinear(torch.nn.Module):
    """Project one-hot class targets into a learned representation.

    Class IDs outside ``[0, num_classes)`` map to an all-zero one-hot vector,
    so their output is the projection bias. This preserves the released TabFM
    checkpoint behavior for padded or otherwise invalid targets.

    Args:
        num_classes: Number of valid class IDs.
        channels: Number of output representation channels.
        device: Device on which to create parameters.
        dtype: Dtype of the parameters.
    """

    def __init__(
        self,
        num_classes: int,
        channels: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        if channels <= 0:
            raise ValueError("channels must be positive")

        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}
        self.num_classes = num_classes
        self.projection = Linear(
            num_classes,
            channels,
            **factory_kwargs,
        )

    def forward(self, target: Tensor) -> Tensor:
        """Encode integer class targets.

        Args:
            target: Class targets with shape ``[..., T]``.

        Returns:
            Target representations with shape ``[..., T, D]``.
        """
        target = target.long()
        valid = (target >= 0) & (target < self.num_classes)
        invalid_class = target.new_full((), self.num_classes)
        mapped = torch.where(valid, target, invalid_class)
        one_hot = F.one_hot(mapped, self.num_classes + 1)
        one_hot = one_hot[..., : self.num_classes]
        return self.projection(one_hot.to(self.projection.weight.dtype))


class ICLearning(torch.nn.Module):
    """Perform label-conditioned in-context learning across table rows.

    Encoded targets are added only to context-row representations. Every row
    may be a query, but attention keys and values are restricted to context
    rows, and row positions do not receive rotary embeddings.

    Args:
        channels: Number of row-representation channels.
        num_blocks: Number of dataset-wise attention blocks.
        num_heads: Number of attention heads.
        max_classes: Maximum number of classification targets.
        feedforward_channels: Hidden width of each transformer feed-forward
            network.
        decoder_hidden: Hidden width of the target encoder and prediction
            decoder MLPs.
        is_classifier: Whether to emit class logits. When ``False``, emit one
            scalar regression value per row.
        activation: Feed-forward activation used by the transformer blocks.
        device: Device on which to create parameters.
        dtype: Dtype of the parameters.
    """

    def __init__(
        self,
        channels: int,
        num_blocks: int,
        num_heads: int,
        max_classes: int,
        feedforward_channels: int,
        decoder_hidden: int,
        is_classifier: bool = True,
        activation: str = "swiglu",
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if max_classes <= 0:
            raise ValueError("max_classes must be positive")
        if decoder_hidden <= 0:
            raise ValueError("decoder_hidden must be positive")

        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}
        self.tf_icl = Encoder(
            num_blocks=num_blocks,
            channels=channels,
            num_heads=num_heads,
            feedforward_channels=feedforward_channels,
            activation=activation,
            rope_theta=None,
            **factory_kwargs,
        )
        self.ln = RMSNorm(channels, **factory_kwargs)
        self.is_classifier = is_classifier
        self.y_encoder: OneHotAndLinear | MLP
        if is_classifier:
            self.y_encoder = OneHotAndLinear(
                num_classes=max_classes,
                channels=channels,
                **factory_kwargs,
            )
            self.decoder = MLP(
                in_channels=channels,
                hidden_channels=[decoder_hidden],
                out_channels=max_classes,
                **factory_kwargs,
            )
        else:
            self.y_encoder = MLP(
                in_channels=1,
                hidden_channels=[decoder_hidden],
                out_channels=channels,
                **factory_kwargs,
            )
            self.decoder = MLP(
                in_channels=channels,
                hidden_channels=[decoder_hidden],
                out_channels=1,
                **factory_kwargs,
            )

    def forward(
        self,
        input: Tensor,
        target: Tensor,
        train_size: Tensor,
        cache: Cache | None = None,
        cache_prefix: str = "icl",
    ) -> Tensor:
        """Predict targets from row representations and context labels.

        Args:
            input: Row representations with shape ``[B, T, D]``.
            target: Padded targets with shape ``[B, T]``. Values at and after
                each table's ``train_size`` do not affect the output.
            train_size: Number of context rows per table with shape ``[B]``.
            cache: Optional record/replay cache for context keys and values.
            cache_prefix: Key namespace used within ``cache``.

        Returns:
            Class logits with shape ``[B, T, K]`` for classification, or scalar
            predictions with shape ``[B, T, 1]`` for regression.
        """
        if input.dim() != 3:
            raise ValueError("input must have shape [B, T, D]")
        if not input.is_floating_point():
            raise ValueError("input must have a floating-point dtype")
        batch_size, num_rows, _ = input.shape
        if num_rows == 0:
            raise ValueError("input must contain at least one row")
        if target.shape != (batch_size, num_rows):
            raise ValueError("target must have shape [B, T]")
        if target.device != input.device:
            raise ValueError("target and input must use the same device")
        if train_size.shape != (batch_size,):
            raise ValueError("train_size must have shape [B]")
        if train_size.is_floating_point():
            raise ValueError("train_size must have an integer dtype")
        if train_size.device != input.device:
            raise ValueError("train_size and input must use the same device")

        mask_key = f"{cache_prefix}.context_mask"
        if cache is not None and cache.is_replaying:
            conditioned = input
            attn_mask = cast(Tensor, cache[mask_key])
        else:
            row_index = torch.arange(num_rows, device=input.device)
            train_mask = row_index[None, :] < train_size[:, None]
            if self.is_classifier:
                target_encoder = cast(OneHotAndLinear, self.y_encoder)
                encoded_target = target_encoder(target)
            else:
                target_encoder = cast(MLP, self.y_encoder)
                encoded_target = target_encoder(
                    target[..., None].to(input.dtype)
                )

            conditioned = input + encoded_target * train_mask[..., None]
            attn_mask = train_mask[:, None, None, :]  # [B, 1, 1, T].
            if cache is not None and cache.is_recording:
                cache[mask_key] = attn_mask
        output = self.tf_icl(
            conditioned,
            attn_mask=attn_mask,
            cache=cache,
            cache_prefix=cache_prefix,
        )
        return self.decoder(self.ln(output))
