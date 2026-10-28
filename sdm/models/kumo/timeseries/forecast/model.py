# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D205

from __future__ import annotations

from typing import Any, ClassVar

import torch
from torch import Tensor

from sdm import Recipe, RelatedTables, Stype, TableTensor, Task
from sdm._warnings import warn_once
from sdm.cache import Cache
from sdm.models import ICLModel
from sdm.models._huggingface import download_checkpoint
from sdm.models.kumo.timeseries.forecast.ckpt import remap_ckpt
from sdm.models.kumo.timeseries.forecast.cross import CrossChannelAttention
from sdm.models.kumo.timeseries.forecast.darr import embed, knn
from sdm.models.kumo.timeseries.forecast.icl import ICLBlock
from sdm.models.kumo.timeseries.forecast.recipe import default_recipe
from sdm.nn import RevIN, SinusoidalEmbedding


class KumoForecast(ICLModel):
    r"""The time series forecasting foundation model from the
    `"Memory-Augmented Forecasting: Scalability and Generalization Across
    Temporal Domains"
    <https://ieeexplore.ieee.org/abstract/document/11402442>`__ paper.

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
    Forecast horizons beyond 72 steps are extended autoregressively, up to
    512 steps.

    For longer observation histories, Domain-Aware Representation and
    Retrieval (DARR) can augment the direct forecast with patterns retrieved
    from historical data. Set ``alpha < 1`` to retrieve similar historical
    windows using nearest neighbors and blend their observed continuations
    with the direct forecast.

    Within the :class:`~sdm.models.ICLModel` protocol, :class:`KumoForecast`
    treats rows as ordered time steps. In a default forward pass,
    ``x_context`` contains past-only historical covariates, ``y_context``
    contains one or more past target series, and ``x_query`` has no columns.
    The number of query rows determines the forecast horizon.

    Args:
        cross_channel: Whether to mix variates after encoding.
        pretrained: Whether to load the pretrained checkpoint.
        device: The device.

    Additional Forward Parameters:
        alpha (float): Blend weight between direct and retrieved forecasts.
            ``1.0`` uses only the direct forecast, ``0.0`` uses only retrieval.
            (default: ``1.0``)
        k (int): Maximum number of historical neighbors to retrieve.
            (default: ``64``)
        temperature (float | None): Softmax temperature for weighting retrieved
            neighbors. (default: ``0.05``)
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

        path = download_checkpoint(
            repo_id="nvidia/Kumo-Forecast",
            filename="run8_best_model_cr.pt"
            if use_cross_channel
            else "moment_head_512_6hr.pt",
            revision="abff20a58834638b28227ff4ab934f26206e4b09",
        )
        ckpt = torch.load(path, map_location=device, weights_only=True)
        self.model.load_state_dict(remap_ckpt(ckpt), assign=True)

        return self

    def forward(
        self,
        x_context: Tensor | TableTensor,
        y_context: Tensor | TableTensor,
        x_query: Tensor | TableTensor,
        *args: Any,
        alpha: float = 1.0,
        k: int = 64,
        temperature: float | None = 0.05,
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

        kwargs.update(alpha=alpha, k=k, temperature=temperature)
        return super().forward(x_context, y_context, x_query, *args, **kwargs)

    def fit(self, *args: Any, **kwargs: Any) -> None:
        r""":meta private:"""  # noqa: D415
        raise NotImplementedError(
            f"{self.__class__.__name__!r} does not support `fit()`"
        )

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
        alpha: float = 1.0,
        k: int = 64,
        temperature: float | None = 0.05,
        batch_size: int = 8,
        **kwargs: Any,
    ) -> TableTensor:
        assert y_context is not None
        assert x_query is not None

        if not 0 <= alpha <= 1:
            raise ValueError(
                f"Expected 'alpha' between zero and one (got {alpha})"
            )

        if y_context.size(-2) < self.model.context_size:
            raise ValueError(
                f"Expected at least {self.model.context_size} context rows "
                f"(got {y_context.size(-2)})"
            )

        horizon = x_query.size(-2)
        if not 1 <= horizon <= 512:
            raise ValueError(
                "Number of query rows must be between 1 and 512 steps "
                f"(got {horizon})"
            )

        context_size = self.model.context_size
        prediction_size = self.model.prediction_size
        memory_horizon = max(prediction_size, horizon)
        use_darr = alpha < 1 and y_context.size(-2) >= (
            self.model.context_size + memory_horizon
        )

        x = y_context.numerical.transpose(-1, -2)  # [..., C, R]
        if x_context_only is not None and x_context_only.size(-1) > 0:
            if self.model.cross_channel is None and not use_darr:
                warn_once(
                    key="kumo-forecast-historical-covariate",
                    message=(
                        "Historical covariates have no effect when "
                        "'cross_channel' and DARR retrieval are inactive."
                    ),
                )
            x = torch.cat(
                [x, x_context_only.numerical.transpose(-1, -2)],
                dim=-2,
            )

        # Forecast:
        last_x = x[..., -self.model.context_size :]
        encoded, mean, scale = self.model.encode(last_x)  # [..., C, P, D]
        out = self.model.decode(encoded, mean, scale)

        # Autoregressive Forecasts:
        if horizon > prediction_size and (alpha != 0 or not use_darr):
            outs = [out]
            for _ in range(prediction_size, horizon, prediction_size):
                last_x = last_x[..., prediction_size:]
                last_x = torch.cat([last_x, out], dim=-1)
                out = self.model(last_x)
                outs.append(out)
            out = torch.cat(outs, dim=-1)
        out = out[..., :horizon]

        if use_darr:  # Memory-based Retrieval:
            key = self.model.encode_memory(
                x=x,
                horizon=memory_horizon,
                batch_size=batch_size,
            )  # [..., N, D]
            query = embed(encoded)  # [..., D]

            window = x.unfold(
                dimension=-1,
                size=context_size + memory_horizon,
                step=prediction_size,
            ).transpose(-3, -2)  # [..., N, C, context_size+memory_horizon]

            retrieved = knn(
                query=query,
                key=key,
                value=window[..., context_size : context_size + horizon],
                k=k,
                temperature=temperature,
            )
            if alpha == 0:
                out = retrieved
            else:
                out = alpha * out + (1 - alpha) * retrieved

        out = out[..., : len(y_context.numerical_columns), :].transpose(-1, -2)
        return TableTensor(
            columns={Stype.numerical: y_context.numerical_columns},
            numerical=out,
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
        encoded, mean, scale = self.encode(x)
        return self.decode(encoded, mean, scale)

    def decode(self, encoded: Tensor, mean: Tensor, scale: Tensor) -> Tensor:
        return self.revin.inverse(self.head(encoded.flatten(-2)), mean, scale)

    def encode_memory(
        self,
        x: Tensor,  # [..., C, R]
        horizon: int,
        batch_size: int,
    ) -> Tensor:  # [..., N, D]
        window = x.unfold(
            dimension=-1,
            size=self.context_size + horizon,
            step=self.prediction_size,
        ).transpose(-3, -2)  # [..., N, C, L + H]

        key = x.new_empty(
            (*x.size()[:-2], window.size(-3), self.lin.out_features),
            dtype=torch.float32,
        )
        for start in range(0, window.size(-3), batch_size):
            x = window[
                ...,
                start : start + batch_size,
                :,
                : self.context_size,
            ]
            x, _, _ = self.encode(x)
            key[..., start : start + batch_size, :] = embed(x)

        return key

    def encode(
        self,
        x: Tensor,  # [..., C, R]
    ) -> tuple[Tensor, Tensor, Tensor]:
        mean, scale = self.revin.compute_stats(x)
        x = self.revin(x, mean, scale)
        x = x.nan_to_num(nan=0, posinf=0, neginf=0)
        x = x.unflatten(-1, (-1, self.patch_size))  # [..., C, N, P]
        x = self.pe(self.lin(x))  # [..., C, N, D]

        x = self.icl(x)  # [..., C, N, D]
        if self.cross_channel is not None:
            x = self.cross_channel(x)  # [..., C, N, D]

        return x, mean, scale
