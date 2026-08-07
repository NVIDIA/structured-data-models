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
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear

from sdm.models.tabfm.attention import _Encoder, _RMSNorm
from sdm.models.tabfm.mlp import _MLP


class _OneHotAndLinear(torch.nn.Module):
    """Project class labels while mapping invalid labels to the bias."""

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
        self.num_classes = num_classes
        self.projection = Linear(
            num_classes,
            channels,
            device=device,
            dtype=dtype,
        )

    def forward(self, targets: Tensor) -> Tensor:
        """Project targets with shape ``[..., T]`` into channels."""
        indices = targets.long()
        valid = indices.ge(0) & indices.lt(self.num_classes)
        indices = indices.where(valid, self.num_classes)
        one_hot = F.one_hot(
            indices,
            num_classes=self.num_classes + 1,
        )[..., : self.num_classes]
        return self.projection(one_hot.to(self.projection.weight.dtype))


class _ICLearning(torch.nn.Module):
    """Apply TabFM v1.0.0 in-context prediction without caching.

    Args:
        channels: Number of representation channels.
        num_blocks: Number of in-context attention blocks.
        num_heads: Number of attention heads.
        feedforward_channels: Hidden width of each attention feed-forward
            layer.
        decoder_hidden_channels: Hidden width of the target encoder and decoder
            MLPs.
        is_classifier: Whether to predict classification logits or regression.
        max_classes: Maximum number of classification classes.
        ffn_chunk_size: Optional maximum tokens per feed-forward chunk.
        device: Device on which to create parameters.
        dtype: Dtype of parameters.
    """

    def __init__(
        self,
        channels: int,
        num_blocks: int,
        num_heads: int,
        feedforward_channels: int,
        decoder_hidden_channels: int,
        is_classifier: bool,
        max_classes: int = 10,
        ffn_chunk_size: int | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}
        self.tf_icl = _Encoder(
            num_blocks=num_blocks,
            channels=channels,
            num_heads=num_heads,
            feedforward_channels=feedforward_channels,
            rope_theta=None,
            ffn_chunk_size=ffn_chunk_size,
            **factory_kwargs,
        )
        self.ln = _RMSNorm(channels, eps=1e-6, **factory_kwargs)
        self.is_classifier = is_classifier
        self.y_encoder: _OneHotAndLinear | _MLP
        if is_classifier:
            self.y_encoder = _OneHotAndLinear(
                num_classes=max_classes,
                channels=channels,
                **factory_kwargs,
            )
            self.decoder = _MLP(
                in_channels=channels,
                hidden_channels=(decoder_hidden_channels,),
                out_channels=max_classes,
                **factory_kwargs,
            )
        else:
            self.y_encoder = _MLP(
                in_channels=1,
                hidden_channels=(decoder_hidden_channels,),
                out_channels=channels,
                **factory_kwargs,
            )
            self.decoder = _MLP(
                in_channels=channels,
                hidden_channels=(decoder_hidden_channels,),
                out_channels=1,
                **factory_kwargs,
            )

    def forward(
        self,
        representations: Tensor,
        targets: Tensor,
        context_size: Tensor,
    ) -> Tensor:
        """Predict every row from representations and context targets."""
        batch_size, num_rows, channels = representations.shape
        if not representations.is_floating_point():
            raise ValueError(
                "representations must be a floating-point [B, T, D] tensor"
            )
        if channels != self.ln.weight.numel():
            raise ValueError(
                "the representation width must match the configured channels"
            )
        if targets.shape != (batch_size, num_rows):
            raise ValueError("targets must have shape [B, T]")
        if (
            context_size.shape != (batch_size,)
            or context_size.is_floating_point()
            or context_size.is_complex()
            or context_size.dtype == torch.bool
            or context_size.device != representations.device
        ):
            raise ValueError(
                "context_size must be an integer [B] tensor on the "
                "representation device"
            )

        row_index = torch.arange(num_rows, device=representations.device)
        context = row_index[None, :] < context_size[:, None]
        clean_targets = targets.where(context, 0)
        if self.is_classifier:
            assert isinstance(self.y_encoder, _OneHotAndLinear)
            encoded_targets = self.y_encoder(clean_targets)
        else:
            assert isinstance(self.y_encoder, _MLP)
            encoded_targets = self.y_encoder(
                clean_targets[..., None].to(representations.dtype)
            )
        representations = torch.where(
            context[..., None],
            representations + encoded_targets,
            representations,
        )
        encoded = self.tf_icl(
            representations,
            attn_mask=context[:, None, :],
        )
        return self.decoder(self.ln(encoded))
