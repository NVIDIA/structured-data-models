# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D205

from typing import Any, ClassVar

import torch
from torch import Tensor

from sdm import Recipe, RelatedTables, Stype, TableTensor, Task
from sdm.cache import Cache
from sdm.models import ICLModel
from sdm.models._huggingface import download_checkpoint
from sdm.models.kumo.timeseries.forecast.recipe import default_recipe


class KumoForecast(ICLModel):
    r"""The time series forecasting foundation model from
    `"New NVIDIA NV-Tesseract Time Series Models Advance Dataset Processing and
    Anomaly Detection" <https://developer.nvidia.com/blog/
    new-nvidia-nv-tesseract-time-series-models-advance-dataset-processing-and-
    anomaly-detection>`__.

    .. figure:: /images/kumo_forecast_light.png
        :width: 100%

    :class:`KumoForecast` is a time-series foundation model for multivariate
    forecasting, with optional historical covariates. Architecturally, it
    combines patch-based time-series embeddings with a
    `MOMENT <https://arxiv.org/abs/2402.03885>`__ transformer encoder and a
    pretrained forecasting head.
    Optional cross-channel mixing lets forecasts use dependencies between
    variates after temporal encoding. The forecasting head predicts the native
    up to 72-step horizon directly from a 512-step history.

    Within the :class:`~sdm.models.ICLModel` protocol, :class:`KumoForecast`
    treats rows as ordered time steps. In a default forward pass,
    ``x_context`` contains past-only historical covariates, ``y_context``
    contains one or more past target series, and ``x_query`` is required to be
    empty.

    Args:
        pretrained: Whether to load the pretrained checkpoint.
        cross_channel: Whether to mix variates after encoding.
        device: The device.
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
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__(task=Task.regression)
        self.model = _KumoForecast(
            cross_channel=cross_channel,
            device="meta" if pretrained else device,
        )
        if pretrained:
            self._load_from_pretrained(cross_channel, device)
        self.eval()

    @classmethod
    def default_recipe(cls) -> Recipe:
        r""":meta private:"""  # noqa: D415
        return default_recipe()

    def _load_from_pretrained(
        self,
        use_cross_channel: bool,
        device: torch.device | str | None,
    ) -> None:
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
