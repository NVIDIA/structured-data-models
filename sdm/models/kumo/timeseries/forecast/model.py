# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D205

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar, Literal

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
from sdm.models.kumo.timeseries.forecast.missingness.backbone import (
    AdaptiveRevIN,
    MissingnessAdapter,
    ObservationAttentionBias,
)
from sdm.models.kumo.timeseries.forecast.missingness.backbone import (
    CrossChannelAttention as MissingnessCrossChannelAttention,
)
from sdm.models.kumo.timeseries.forecast.missingness.ckpt import (
    remap_ckpt as remap_missingness_ckpt,
)
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
    Forecast horizons beyond 72 steps are extended autoregressively.

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
        missingness: Whether to select the missingness-aware model variant
            (using 96-step context and 24-step forecast blocks).
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
        channel_alignment (str | None): Channel alignment for
            ``missingness=True``. ``'name'`` maps auxiliary columns to the
            checkpoint's ``HUFL, HULL, LUFL, LULL, MUFL, MULL`` slots and masks
            absent channels. The first target always occupies slot zero.
            ``'positional'`` assigns slots in target-then-covariate order.
            Defaults to name alignment for TableTensor context inputs and
            positional alignment for Tensor context inputs.
        history_rows (int | None): For ``missingness=True``, use only the last
            this many context rows, including for fitting normalization.
            Must be at least 96. Defaults to all supplied context rows.
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

    paper: ClassVar[str] = (
        "https://ieeexplore.ieee.org/abstract/document/11402442"
    )
    source: ClassVar[str] = "https://github.com/NVIDIA/Kumo-TS"
    release_date: ClassVar[str] = "2026-04-07 to 2026-11-10"
    parameter_count: ClassVar[str] = "313.06M-317.26M"
    code_license: ClassVar[tuple[str, str]] = (
        "Apache-2.0",
        "https://github.com/NVIDIA/structured-data-models/blob/main/LICENSE",
    )
    weights_license: ClassVar[tuple[str, str]] = (
        "Apache-2.0",
        "https://huggingface.co/nvidia/Kumo-Forecast",
    )
    metadata: ClassVar[Mapping[str, str]] = {
        "Output": "Point predictions per target and time step",
        "Missing Values": (
            "Handled natively with ``missingness=True``, zero imputation in "
            "the default recipe otherwise"
        ),
        "Pretraining Scope": "Up to 512 rows",
    }
    channel_schema: ClassVar[tuple[str, ...]] = (
        "OT",
        "HUFL",
        "HULL",
        "LUFL",
        "LULL",
        "MUFL",
        "MULL",
    )

    def __init__(
        self,
        cross_channel: bool = True,
        missingness: bool = False,
        pretrained: bool = True,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__(task=Task.regression)
        self.cross_channel = cross_channel

        if missingness and not self.cross_channel:
            self.cross_channel = True
            warn_once(
                key="kumo-forecast-missingness-and-cross-channel",
                message=(
                    f"`cross_channel=False` will be ignored in "
                    f"{self.__class__.__name__}(missingness=True)."
                ),
            )

        self.model: _KumoForecast | _KumoForecastMissingness
        if missingness:
            self.model = _KumoForecastMissingness(
                in_channels=len(self.channel_schema),
                device="meta" if pretrained else device,
            )
        else:
            self.model = _KumoForecast(
                cross_channel=cross_channel,
                device="meta" if pretrained else device,
            )

        if pretrained:
            self._load_from_pretrained(
                cross_channel=cross_channel,
                missingness=missingness,
                device=device,
            )

        self.eval()

    def default_recipe(self) -> Recipe:
        r""":meta private:"""  # noqa: D415
        missingness = isinstance(self.model, _KumoForecastMissingness)
        return default_recipe(missingness)

    def _load_from_pretrained(
        self,
        cross_channel: bool,
        missingness: bool,
        device: torch.device | str | None,
    ) -> KumoForecast:
        device = torch.get_default_device() if device is None else device

        if missingness:
            path = download_checkpoint(
                repo_id="nvidia/Kumo-Forecast",
                filename="kumo-forecast-1.2.0/best_model.pt",
                revision="402caa11b7ff585141e912c6743026141f648892",
            )
            ckpt = torch.load(
                path, map_location="cpu", weights_only=True, mmap=True
            )
            self.model.load_state_dict(
                remap_missingness_ckpt(ckpt["model_state_dict"]), assign=True
            )
            self.model.to(device)
            return self

        path = download_checkpoint(
            repo_id="nvidia/Kumo-Forecast",
            filename="run8_best_model_cr.pt"
            if cross_channel
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
        channel_alignment: Literal["name", "positional"] | None = None,
        history_rows: int | None = None,
        **kwargs: Any,
    ) -> TableTensor:
        r""":meta private:"""  # noqa: D415
        if isinstance(self.model, _KumoForecastMissingness):
            if channel_alignment is None:
                channel_alignment = (
                    "name"
                    if isinstance(x_context, TableTensor)
                    or isinstance(y_context, TableTensor)
                    else "positional"
                )
            kwargs["channel_alignment"] = channel_alignment
            if history_rows is not None:
                if history_rows < self.model.context_size:
                    raise ValueError("Expected history_rows >= 96")
                x_context = x_context[..., -history_rows:, :]
                y_context = y_context[..., -history_rows:, :]
        if not isinstance(x_context, TableTensor):
            x_context = TableTensor.from_tensor(x_context)
        if not isinstance(x_query, TableTensor):
            x_query = TableTensor.from_tensor(x_query)
        if isinstance(self.model, _KumoForecastMissingness):
            if x_query.size(-1):
                raise ValueError(
                    "Missingness-aware forecasting requires a query with "
                    "zero columns (no future covariates)"
                )
            if not 1 <= x_query.size(-2) <= 512:
                raise ValueError(
                    "Expected a forecast horizon between 1 and 512 rows"
                )
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
        channel_alignment: Literal["name", "positional"] = "positional",
        **kwargs: Any,
    ) -> TableTensor:
        assert y_context is not None
        assert x_query is not None

        if not 0 <= alpha <= 1:
            raise ValueError(
                f"Expected 'alpha' between zero and one (got {alpha})"
            )

        if isinstance(self.model, _KumoForecastMissingness) and alpha != 1.0:
            raise ValueError(
                f"DARR is not supported in "
                f"{self.__class__.__name__}(missingness=True)"
            )

        if y_context.size(-2) < self.model.context_size:
            raise ValueError(
                f"Expected at least {self.model.context_size} context rows "
                f"(got {y_context.size(-2)})"
            )
        if isinstance(self.model, _KumoForecastMissingness) and (
            not y_context.numerical[..., -self.model.context_size :, 0]
            .isfinite()
            .any(-1)
            .all()
        ):
            raise ValueError(
                "The first target must have observed values "
                "in the last 96 rows"
            )

        horizon = x_query.size(-2)
        context_size = self.model.context_size
        prediction_size = self.model.prediction_size
        memory_horizon = max(prediction_size, horizon)
        use_darr = alpha < 1 and y_context.size(-2) >= (
            self.model.context_size + memory_horizon
        )

        columns = list(y_context.numerical_columns)
        x = y_context.numerical.transpose(-1, -2)  # [..., C, R]
        if x_context_only is not None and x_context_only.size(-1) > 0:
            columns.extend(x_context_only.numerical_columns)
            if not self.cross_channel and not use_darr:
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
        valid = indices = None
        if isinstance(self.model, _KumoForecastMissingness):
            last_x = last_x.to(self.model.lin.weight.dtype)
            if channel_alignment == "name":
                if len(set(columns)) != len(columns) or any(
                    name not in self.channel_schema[1:] for name in columns[1:]
                ):
                    raise ValueError(
                        "Named auxiliary channels must be unique and belong "
                        f"to {self.channel_schema[1:]}; use "
                        "channel_alignment='positional' for other names"
                    )
                slots = [0] + [
                    self.channel_schema.index(name) for name in columns[1:]
                ]
                indices = torch.tensor(slots, device=x.device)
                history = last_x.new_full(
                    (
                        *last_x.shape[:-2],
                        len(self.channel_schema),
                        context_size,
                    ),
                    torch.nan,
                )
                history.index_copy_(-2, indices, last_x)
                last_x = history
                valid = torch.zeros(
                    len(self.channel_schema), dtype=torch.bool, device=x.device
                )
                valid[indices] = True
            elif channel_alignment != "positional":
                raise ValueError(
                    f"Unknown channel alignment: {channel_alignment!r}"
                )

        if use_darr:
            assert isinstance(self.model, _KumoForecast)
            encoded, mean, scale = self.model.encode(last_x)  # [..., C, P, D]
            out = self.model.decode(encoded, mean, scale)
        elif isinstance(self.model, _KumoForecastMissingness):
            out = self.model(last_x, valid)
        else:
            out = self.model(last_x)

        # Autoregressive Forecasts:
        if horizon > prediction_size and (alpha != 0 or not use_darr):
            outs = [out]
            for _ in range(prediction_size, horizon, prediction_size):
                if valid is not None:
                    out = out.masked_fill(~valid.unsqueeze(-1), torch.nan)
                last_x = last_x[..., prediction_size:]
                last_x = torch.cat([last_x, out], dim=-1)
                if isinstance(self.model, _KumoForecastMissingness):
                    out = self.model(last_x, valid)
                else:
                    out = self.model(last_x)
                outs.append(out)
            out = torch.cat(outs, dim=-1)
        out = out[..., :horizon]

        if use_darr:  # Memory-based Retrieval:
            assert isinstance(self.model, _KumoForecast)
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

        if indices is not None:
            out = out.index_select(-2, indices[: y_context.size(-1)])
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
        out = self.head(encoded.flatten(-2)).float()
        return self.revin.inverse(out, mean, scale)

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
        mean, scale = self.revin.stats(x, correction=0)
        x = self.revin(x, mean, scale)
        x = x.nan_to_num(nan=0, posinf=0, neginf=0)
        x = x.unflatten(-1, (-1, self.patch_size))  # [..., C, N, P]
        x = self.pe(self.lin(x))  # [..., C, N, D]

        x = self.icl(x)  # [..., C, N, D]
        if self.cross_channel is not None:
            x = self.cross_channel(x)  # [..., C, N, D]

        return x, mean, scale


class _KumoForecastMissingness(torch.nn.Module):
    def __init__(
        self,
        context_size: int = 96,
        prediction_size: int = 24,
        patch_size: int = 8,
        in_channels: int = 7,
        channels: int = 1024,
        hidden_channels: int = 2816,
        num_layers: int = 24,
        num_heads: int = 16,
        cross_channel_heads: int = 8,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        self.context_size = context_size
        self.prediction_size = prediction_size
        self.patch_size = patch_size

        self.normalizer = AdaptiveRevIN(in_channels, device=device)
        self.mask_token = torch.nn.Parameter(
            torch.zeros(1, 1, 1, device=device)
        )
        self.mask_embedding = torch.nn.Parameter(
            torch.zeros(channels, device=device)
        )
        self.lin = torch.nn.Linear(
            patch_size, channels, bias=False, device=device
        )
        self.pe = SinusoidalEmbedding()
        self.obs_attn_bias = ObservationAttentionBias(num_heads, device=device)
        self.icl = ICLBlock(
            channels=channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            num_heads=num_heads,
            num_buckets=32,
            max_distance=128,
            device=device,
        )
        self.cross_channel = MissingnessCrossChannelAttention(
            channels=channels,
            num_heads=cross_channel_heads,
            num_variates=in_channels,
            device=device,
        )
        self.missingness_adapter = MissingnessAdapter(channels, device=device)
        self.head = torch.nn.Linear(
            context_size // patch_size * channels,
            prediction_size,
            device=device,
        )
        if dtype is not None:
            self.to(dtype=dtype)

    def forward(
        self,
        x: Tensor,  # [..., C, R]
        valid: Tensor | None = None,  # [..., C]
    ) -> Tensor:  # [..., C, prediction_size]
        if valid is None:
            valid = torch.ones(x.size(-2), dtype=torch.bool, device=x.device)
        observed = x.isfinite() & valid.unsqueeze(-1)
        x = x.masked_fill(~observed, 0)
        mean, scale = self.normalizer(x, observed, valid)
        x = (
            ((x - mean) / scale)
            .nan_to_num(nan=0, posinf=0, neginf=0)
            .clamp(-10, 10)
        )
        x = x + ~observed * self.mask_token.reshape(())
        patches = x.unflatten(-1, (-1, self.patch_size))
        patch_observed = observed.unflatten(-1, (-1, self.patch_size)).any(-1)
        x = self.lin(patches)
        x = torch.where(patch_observed.unsqueeze(-1), x, self.mask_embedding)
        x = self.pe(x)
        bias = self.obs_attn_bias(patch_observed).to(x.dtype)
        # T5 uses finite minimum logits even for entirely missing channels.
        bias = bias.masked_fill(
            ~patch_observed[..., None, None, :], torch.finfo(x.dtype).min
        )
        x = self.icl(x, attn_mask=bias)
        x = self.cross_channel(x, patch_observed, valid)
        x = self.missingness_adapter(x, patch_observed, valid)
        return self.head(x.flatten(-2)) * scale + mean
