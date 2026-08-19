from typing import Any, cast

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import GELU, Linear, ModuleList, RMSNorm, Sequential

from sdm.cache import Cache, KVCacheEntry
from sdm.models.kumo.tabular.block import KumoTabularTransformerBlock


class ICLBlock(torch.nn.Module):
    """Predict query targets from context row representations.

    Args:
        num_classes: Number of context target classes.
        out_channels: Number of output channels per query row.
        channels: Number of channels per row representation.
        num_layers: Number of in-context transformer layers.
        num_heads: Number of attention heads.
        device: Device of the parameters.
        dtype: Data type of the parameters.
    """

    def __init__(
        self,
        num_classes: int,
        out_channels: int,
        channels: int,
        num_layers: int,
        num_heads: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}

        self.num_classes = num_classes
        self.y_lin = Linear(num_classes, channels, **factory_kwargs)
        self.layers = ModuleList(
            KumoTabularTransformerBlock(
                channels=channels,
                num_heads=num_heads,
                **factory_kwargs,
            )
            for _ in range(num_layers)
        )
        self.norm = RMSNorm(channels, **factory_kwargs)
        self.head = Sequential(
            Linear(channels, 2 * channels, **factory_kwargs),
            GELU(),
            Linear(2 * channels, out_channels, **factory_kwargs),
        )

    def forward(
        self,
        x: Tensor,  # [..., R, D]
        y: Tensor,  # [..., R_train]
        *,
        cache: Cache | None = None,
        batch_size_limit: int | None = None,
    ) -> Tensor:  # [..., R_test, out_channels]
        """Predict targets for the query rows.

        Args:
            x: Context-first row representations with shape ``[..., R, D]``,
                where ``R`` is the number of rows and ``D`` is the number of
                channels.
            y: Context labels with shape ``[..., R_context]``.
            cache: Optional key/value cache for context reuse.
            batch_size_limit: Optional attention batch-size limit.

        Returns:
            Query logits with shape ``[..., R_query, O]``, where ``O`` is
            ``out_channels``.
        """
        num_train = y.size(-1)
        y_one_hot = F.one_hot(
            y.long(),
            num_classes=self.num_classes,
        ).to(self.y_lin.weight.dtype)
        y_emb = self.y_lin(y_one_hot).to(x.dtype)
        x = torch.cat(
            [x[..., :num_train, :] + y_emb, x[..., num_train:, :]],
            dim=-2,
        )

        for i, layer in enumerate(self.layers):
            key = f"icl_block.layer{i}"
            result = layer(
                query=(
                    x[..., num_train:, :] if i == len(self.layers) - 1 else x
                ),
                key_value=(
                    cast(KVCacheEntry, cache[key])
                    if cache is not None and cache.is_replaying
                    else x[..., :num_train, :]
                ),
                return_key_value=cache is not None and cache.is_recording,
                batch_size_limit=batch_size_limit,
            )
            if cache is not None and cache.is_recording:
                x, cache[key] = result
            else:
                x = result

        return self.head(self.norm(x))
