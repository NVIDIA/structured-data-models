# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Any, ClassVar

import torch
from torch import Tensor

from sdm import Recipe, RelatedTables, Stype, TableTensor, Task
from sdm.cache import Cache
from sdm.models import ICLModel
from sdm.models._huggingface import download_checkpoint
from sdm.models.kumo.timeseries.forecast.recipe import default_recipe


class KumoForecast(ICLModel):
    """Kumo time-series forecasting with the Kumo-Forecast checkpoints.

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
        cross_channel: Whether to mix variates after the T5 encoder.
            Selects the cross-channel or base published checkpoint.
        device: The device for model parameters.
    """

    supported_feature_stypes: ClassVar[frozenset[Stype]] = frozenset(
        {Stype.numerical}
    )
    supported_target_stypes: ClassVar[frozenset[Stype]] = frozenset(
        {Stype.numerical}
    )
    supports_multi_target: ClassVar[bool] = True
    supports_context_only_columns: ClassVar[bool] = True
    supports_related_tables: ClassVar[bool] = False

    def __init__(
        self,
        pretrained: bool = True,
        cross_channel: bool = True,
        checkpoint: str | Path | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__(task=Task.regression)
        load_weights = pretrained or checkpoint is not None
        self.model = _KumoForecast(
            cross_channel=cross_channel,
            device="meta" if load_weights else device,
        )
        if load_weights:
            self._load_from_pretrained(checkpoint, cross_channel, device)
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
                repo_id="nvidia/Kumo-Forecast",
                filename="run8_best_model_cr.pt"
                if use_cross_channel
                else "moment_head_512_6hr.pt",
                revision="abff20a58834638b28227ff4ab934f26206e4b09",
            )
        device = torch.get_default_device() if device is None else device
        torch.load(checkpoint, map_location=device, weights_only=True)

    def _forward(
        self,
        x_context: TableTensor | None,
        y_context: TableTensor | None,
        x_query: TableTensor | None,
        x_context_only: TableTensor | None,  # [..., R_context, D_only]
        related_context_tables: RelatedTables[TableTensor] | None,
        related_query_tables: RelatedTables[TableTensor] | None,
        cache: Cache | None,
        generator: torch.Generator | None,
        **kwargs: Any,
    ) -> TableTensor:
        raise NotImplementedError


class _KumoForecast(torch.nn.Module):
    def __init__(
        self,
        cross_channel: bool,
        context_length: int = 512,
        prediction_length: int = 72,
        patch_len: int = 8,
        channels: int = 1024,
        hidden_channels: int = 2816,
        num_layers: int = 24,
        num_heads: int = 16,
        head_channels: int = 64,
        cross_channel_heads: int = 8,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        return x
