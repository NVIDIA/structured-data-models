# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar, cast

import torch
from torch import Tensor

from sdm import Recipe, RelatedTables, Stype, TableTensor, Task
from sdm.cache import Cache
from sdm.models import ICLModel
from sdm.models._huggingface import download_checkpoint
from sdm.models.kumo.timeseries.forecasting.attention import (
    CrossChannelAttention,
)
from sdm.models.kumo.timeseries.forecasting.ckpt import remap_ckpt
from sdm.models.kumo.timeseries.forecasting.encoder import T5Encoder
from sdm.models.kumo.timeseries.forecasting.head import ForecastingHead
from sdm.models.kumo.timeseries.forecasting.normalization import RevIN
from sdm.models.kumo.timeseries.forecasting.patch import (
    PatchEmbedding,
    Patching,
    patch_mask,
)
from sdm.models.kumo.timeseries.forecasting.recipe import default_recipe
from sdm.tensor.table import TableSchema

MODEL_KWARGS: dict[str, Any] = {
    "context_length": 512,
    "prediction_length": 72,
    "patch_len": 8,
    "channels": 1024,
    "hidden_channels": 2816,
    "num_layers": 24,
    "num_heads": 16,
    "head_channels": 64,
    "cross_channel_heads": 8,
}


class KumoForecasting(ICLModel):
    """Kumo time-series forecasting with the NV-Tesseract checkpoints.

    Rows are chronological, regularly spaced time steps. ``y_context`` holds
    one or more target histories; numerical ``x_context`` columns are optional
    past covariates. The last 512 rows are used and at least 512 are required.

    ``x_query`` determines a forecast length from 1 to 72 steps. Its values
    are ignored: this model does not support future-known covariates. Supply
    NaN placeholders with the same schema as ``x_context``. With no covariates,
    use tensors of shape ``[T, 0]`` and ``[H, 0]``. Predictions retain target
    column names and original scale. Higher-rank inputs follow SDM's ensemble
    semantics; pass ``num_estimators=1`` to preserve batch dimensions.

    The default recipe reproduces the published fixed standardizer. This is
    direct backbone forecasting; SDK retrieval correction and autoregressive
    extrapolation beyond the checkpoint horizon are not included.

    Args:
        pretrained: Whether to download pretrained weights when ``checkpoint``
            is not supplied.
        use_cross_channel: Whether to mix variates after the T5 encoder.
            Selects the cross-channel or base published checkpoint.
        checkpoint: Optional local NV-Tesseract state-dict file. Architecture
            must match the selected model. Loaded with ``weights_only=True``.
        device: The device for model parameters.
    """

    supported_feature_stypes: ClassVar[frozenset[Stype]] = frozenset(
        {Stype.numerical}
    )
    supported_target_stypes: ClassVar[frozenset[Stype]] = frozenset(
        {Stype.numerical}
    )
    supports_multi_target: ClassVar[bool] = True
    supports_related_tables: ClassVar[bool] = False

    def __init__(
        self,
        pretrained: bool = True,
        use_cross_channel: bool = True,
        checkpoint: str | Path | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__(task=Task.regression)
        load_weights = pretrained or checkpoint is not None
        self.model = _KumoForecasting(
            use_cross_channel=use_cross_channel,
            device="meta" if load_weights else device,
            **MODEL_KWARGS,
        )
        if load_weights:
            self._load_from_pretrained(checkpoint, use_cross_channel, device)
        self.eval()

    @classmethod
    def default_recipe(cls) -> Recipe:
        r""":meta private:"""  # noqa: D415
        return default_recipe()

    def _load_from_pretrained(
        self,
        checkpoint: str | Path | None,
        use_cross_channel: bool,
        device: torch.device | str | None,
    ) -> None:
        if checkpoint is None:
            checkpoint = download_checkpoint(
                repo_id="nvidia/nv-tesseract-forecasting",
                filename="run8_best_model_cr.pt"
                if use_cross_channel
                else "moment_head_512_6hr.pt",
                revision="abff20a58834638b28227ff4ab934f26206e4b09",
            )
        device = torch.get_default_device() if device is None else device
        ckpt = torch.load(checkpoint, map_location=device, weights_only=True)
        self.model.load_state_dict(remap_ckpt(ckpt), strict=True, assign=True)

    def _forward(
        self,
        x_context: TableTensor | None,
        y_context: TableTensor | None,
        x_query: TableTensor | None,
        related_context_tables: RelatedTables[TableTensor] | None,
        related_query_tables: RelatedTables[TableTensor] | None,
        cache: Cache | None,
        generator: torch.Generator | None,
        **kwargs: Any,
    ) -> TableTensor:
        if (
            x_query is not None
            and not 1 <= x_query.size(-2) <= self.model.prediction_length
        ):
            raise ValueError(
                "Forecast length must be between 1 and "
                f"{self.model.prediction_length} steps"
            )

        if cache is not None and cache.is_replaying:
            prediction = cast(Tensor, cache["forecast"])
            columns = cast(TableSchema, cache["y_schema"]).columns[
                Stype.numerical
            ]
        else:
            assert x_context is not None
            assert y_context is not None
            columns = y_context.columns[Stype.numerical]
            if not columns:
                raise ValueError(
                    "Expected at least one numerical target series"
                )
            if x_context.size(-2) < self.model.context_length:
                raise ValueError(
                    f"Expected at least {self.model.context_length} "
                    "history rows"
                )
            # [*batch, T, V] -> [B, V, T]; targets precede past covariates.
            history = torch.cat(
                [y_context.numerical, x_context.numerical], dim=-1
            )
            history = history[..., -self.model.context_length :, :]
            batch_shape = history.shape[:-2]
            history = history.reshape(-1, *history.shape[-2:]).transpose(
                -1, -2
            )
            history = history.to(self.model.head.linear.weight.dtype)
            prediction = self.model(history)[:, : len(columns)].transpose(
                -1, -2
            )
            prediction = prediction.reshape(
                *batch_shape, self.model.prediction_length, len(columns)
            )
            if cache is not None:
                cache["forecast"] = prediction

        if x_query is not None and x_query.shape[:-2] != prediction.shape[:-2]:
            raise ValueError(
                "Expected context and query batch dimensions to match"
            )
        length = 0 if x_query is None else x_query.size(-2)
        return TableTensor(
            columns={Stype.numerical: columns},
            numerical=prediction[..., :length, :],
        )


class _KumoForecasting(torch.nn.Module):
    def __init__(
        self,
        context_length: int = 512,
        prediction_length: int = 72,
        patch_len: int = 8,
        channels: int = 1024,
        hidden_channels: int = 2816,
        num_layers: int = 24,
        num_heads: int = 16,
        head_channels: int = 64,
        use_cross_channel: bool = True,
        cross_channel_heads: int = 8,
        dropout: float = 0.1,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.normalizer = RevIN(num_features=1, affine=False, **factory_kwargs)
        self.tokenizer = Patching(patch_len=patch_len, stride=patch_len)
        self.patch_embedding = PatchEmbedding(
            patch_len=patch_len,
            stride=patch_len,
            channels=channels,
            dropout=dropout,
            add_positional_embedding=True,
            **factory_kwargs,
        )
        self.encoder = T5Encoder(
            channels=channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            num_heads=num_heads,
            head_channels=head_channels,
            dropout=dropout,
            **factory_kwargs,
        )
        if use_cross_channel:
            self.cross_channel_attn = CrossChannelAttention(
                channels=channels,
                num_heads=cross_channel_heads,
                dropout=dropout,
                **factory_kwargs,
            )
        else:
            self.cross_channel_attn = torch.nn.Identity()
        self.head = ForecastingHead(
            input_channels=channels * (context_length // patch_len),
            prediction_length=prediction_length,
            dropout=dropout,
            **factory_kwargs,
        )

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        x, state = self.normalizer(x, mask)
        x = x.nan_to_num(nan=0, posinf=0, neginf=0)
        # Forecasting embeds all patches, even when the attention mask excludes
        # some of them; unlike reconstruction, it does not insert mask tokens.
        x = self.patch_embedding(self.tokenizer(x))
        batch_size, num_variates, num_patches, channels = x.shape
        x = x.reshape(batch_size * num_variates, num_patches, channels)
        if mask is not None:
            mask = patch_mask(
                mask, self.tokenizer.patch_len, self.tokenizer.stride
            )
            mask = mask.repeat_interleave(num_variates, dim=0)
        x = self.encoder(x, mask)
        x = x.reshape(batch_size, num_variates, num_patches, channels)
        x = self.cross_channel_attn(x)
        return self.normalizer.inverse(self.head(x), state)
