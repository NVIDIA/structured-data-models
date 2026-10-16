# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D205

from __future__ import annotations

from typing import Any, ClassVar, cast

import torch
from torch import Tensor

from sdm import Recipe, RelatedTables, Stype, TableTensor, Task
from sdm._warnings import warn_once
from sdm.cache import Cache
from sdm.models import ICLModel
from sdm.models._huggingface import download_checkpoint
from sdm.models.kumo.timeseries.forecast.ckpt import remap_ckpt
from sdm.models.kumo.timeseries.forecast.cross import CrossChannelAttention
from sdm.models.kumo.timeseries.forecast.icl import ICLBlock
from sdm.models.kumo.timeseries.forecast.recipe import default_recipe
from sdm.nn import RevIN, SinusoidalEmbedding
from sdm.tensor.table import TableSchema


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
        cross_channel: Whether to mix variates after encoding.
        pretrained: Whether to load the pretrained checkpoint.
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
        cross_channel: bool = True,
        pretrained: bool = True,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__(task=Task.regression)

        self.model = _KumoForecast(
            cross_channel=cross_channel,
            device="meta" if pretrained else device,
        )

        if pretrained:
            self._load_from_pretrained(cross_channel, device=device)

        self.eval()

    @classmethod
    def default_recipe(cls) -> Recipe:
        r""":meta private:"""  # noqa: D415
        return default_recipe()

    def _load_from_pretrained(
        self,
        use_cross_channel: bool,
        device: torch.device | str | None,
    ) -> KumoForecast:
        device = torch.get_default_device() if device is None else device

        checkpoint = download_checkpoint(
            repo_id="nvidia/Kumo-Forecast",
            filename="run8_best_model_cr.pt"
            if use_cross_channel
            else "moment_head_512_6hr.pt",
            revision="abff20a58834638b28227ff4ab934f26206e4b09",
        )
        ckpt = torch.load(checkpoint, map_location=device, weights_only=True)
        self.model.load_state_dict(remap_ckpt(ckpt), assign=True)

        return self

    def forward(
        self,
        x_context: Tensor | TableTensor,
        y_context: Tensor | TableTensor,
        x_query: Tensor | TableTensor,
        *args: Any,
        **kwargs: Any,
    ) -> TableTensor:
        r""":meta private:"""  # noqa: D415
        if not isinstance(x_context, TableTensor):
            x_context = TableTensor.from_tensor(x_context)
        if not isinstance(x_query, TableTensor):
            x_query = TableTensor.from_tensor(x_query)
        if x_query.size(-1) > 0:
            warn_once(
                key="kumo-forecast-future-covariate",
                message=(
                    f"{self.__class__.__name__!r} does not support future "
                    f"covariates. Query columns will be ignored."
                ),
            )
            x_query = x_query[..., :0]

        return super().forward(x_context, y_context, x_query, *args, **kwargs)

    def fit(self, x: Tensor | TableTensor, *args: Any, **kwargs: Any) -> None:
        r""":meta private:"""  # noqa: D415
        if isinstance(x, TableTensor):
            kwargs["context_only_columns"] = [
                column for columns in x.columns.values() for column in columns
            ]
        else:
            kwargs["context_only_columns"] = range(x.size(-1))

        return super().fit(x, *args, **kwargs)

    def predict(
        self,
        x: Tensor | TableTensor,
        *args: Any,
        **kwargs: Any,
    ) -> TableTensor:
        r""":meta private:"""  # noqa: D415
        if x.size(-1) > 0:
            warn_once(
                key="kumo-forecast-future-covariate",
                message=(
                    f"{self.__class__.__name__!r} does not support future "
                    f"covariates. Query columns will be ignored."
                ),
            )
            x = x[..., :0]

        return super().predict(x, *args, **kwargs)

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

        if (
            y_context is not None
            and y_context.size(-2) < self.model.context_size
        ):
            raise ValueError(
                f"Expected at least {self.model.context_size} context rows "
                f"(got {y_context.size(-2)})"
            )

        if (
            x_query is not None
            and not 1 <= x_query.size(-2) <= self.model.prediction_size
        ):
            raise ValueError(
                "Number of query rows must be between 1 and "
                f"{self.model.prediction_size} steps "
                f"(got {x_query.size(-2)})"
            )

        if cache is None or cache.is_recording:
            assert y_context is not None
            y_schema = y_context.schema
            x = y_context.numerical
            if x_context_only is not None:
                x = torch.cat([x, x_context_only.numerical], dim=-1)

            x = x[..., -self.model.context_size :, :].transpose(-1, -2)
            out = self.model(x)[..., : y_context.size(-1), :].transpose(-1, -2)

            if cache is not None:
                cache["out"] = out
        else:
            y_schema = cast(TableSchema, cache["y_schema"])
            out = cast(Tensor, cache["out"])

        length = 0 if x_query is None else x_query.size(-2)
        return TableTensor(
            columns={Stype.numerical: y_schema.columns[Stype.numerical]},
            numerical=out[..., :length, :],
        )


class _KumoForecast(torch.nn.Module):
    def __init__(
        self,
        cross_channel: bool,
        context_size: int = 512,
        prediction_size: int = 72,
        patch_size: int = 8,
        channels: int = 1024,
        hidden_channels: int = 2816,
        num_layers: int = 24,
        num_heads: int = 16,
        cross_channel_heads: int = 8,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}

        self.context_size = context_size
        self.prediction_size = prediction_size
        self.patch_size = patch_size

        self.revin = RevIN(eps=1e-5, **factory_kwargs)
        self.lin = torch.nn.Linear(
            in_features=patch_size,
            out_features=channels,
            bias=False,
            **factory_kwargs,
        )
        torch.nn.init.orthogonal_(self.lin.weight, gain=1.41)
        self.pe = SinusoidalEmbedding()

        self.icl = ICLBlock(
            channels=channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            num_heads=num_heads,
            num_buckets=32,
            max_distance=128,
            **factory_kwargs,
        )
        self.cross_channel: CrossChannelAttention | None = None
        if cross_channel:
            self.cross_channel = CrossChannelAttention(
                channels=channels,
                num_heads=cross_channel_heads,
                **factory_kwargs,
            )

        self.head = torch.nn.Linear(
            in_features=channels * (context_size // patch_size),
            out_features=prediction_size,
            **factory_kwargs,
        )

    def forward(
        self,
        x: Tensor,  # [..., C, R]
    ) -> Tensor:  # [..., C, prediction_size]

        mean, scale = self.revin.compute_stats(x)
        x = self.revin(x, mean, scale)
        x = x.nan_to_num(nan=0, posinf=0, neginf=0)
        x = x.unflatten(-1, (-1, self.patch_size))  # [..., C, N, P]
        x = self.pe(self.lin(x))  # [..., C, N, D]

        x = self.icl(x)  # [..., C, N, D]
        if self.cross_channel is not None:
            x = self.cross_channel(x)  # [..., C, N, D]

        x = self.head(x.flatten(-2))  # [..., C, prediction_size]
        return self.revin.inverse(x, mean, scale)
