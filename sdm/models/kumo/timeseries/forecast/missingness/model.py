# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Mapping
from typing import Any, ClassVar, Literal

import torch
from torch import Tensor

from sdm import Recipe, RelatedTables, Stype, TableTensor, Task
from sdm.cache import Cache
from sdm.models import ICLModel
from sdm.models._huggingface import download_checkpoint
from sdm.models.kumo.timeseries.forecast.missingness.backbone import (
    MissingnessBackbone,
)
from sdm.models.kumo.timeseries.forecast.missingness.ckpt import remap_ckpt
from sdm.models.kumo.timeseries.forecast.missingness.recipe import (
    default_recipe,
)


class KumoForecastMissingness(ICLModel):
    """Kumo forecasting with explicit observation masks and channel identities.

    This is the missingness-aware model from Kumo-TS v1.0.1, using the separate
    ``kumo-forecast-1.2.0`` checkpoint. It forecasts from the last 96 context
    rows in 24-step blocks, up to 512 steps autoregressively. NaN and infinite
    values are unobserved; observed zeros remain observations.

    Rows must be ordered time steps. ``y_context`` contains the targets and
    ``x_context`` optional historical covariates. ``x_query`` must have no
    columns; its row count specifies the horizon. Predictions have the target
    columns and original units. To predict every channel, put them all in
    ``y_context`` and supply an empty ``x_context``.

    The first target occupies the checkpoint's primary target slot. Named
    inputs align other columns to ``HUFL, HULL, LUFL, LULL, MUFL, MULL``;
    absent channels are masked padding. Tensor inputs use positional order:
    targets followed by covariates. Explicit ``channel_alignment='positional'``
    accepts arbitrary names but assigns learned identities and priors by
    position. Channels beyond seven share the last identity and use the mean
    trained scale prior.

    The default recipe fits observed-value statistics on all supplied context
    rows and freezes them during forecasting. Supply only past data. Custom
    recipes must preserve gaps and channel order. Frozen reference statistics
    can be supplied through :meth:`~sdm.processing.Standardize.with_state`;
    this checkpoint contains no dataset standardizer. DARR and interpretability
    are not supported by this model.

    Args:
        pretrained: Whether to load the released pretrained checkpoint.
        device: The device.

    Additional Forward Parameters:
        channel_alignment (str | None): ``'name'`` or ``'positional'``.
            Defaults to name alignment for TableTensor inputs and positional
            alignment when both context inputs are tensors.
        history_rows (int | None): Use only the last this many context rows,
            including for fitting normalization. Must be at least 96.
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

    source: ClassVar[str] = "https://github.com/NVIDIA/Kumo-TS/tree/v1.0.1"
    paper: ClassVar[str] = (
        "https://ieeexplore.ieee.org/abstract/document/11402442"
    )
    release_date: ClassVar[str] = "2026-11-10"
    parameter_count: ClassVar[str] = "313.37M"
    metadata: ClassVar[Mapping[str, str]] = {
        "Output": "Point predictions in original target units",
        "Missing Values": "Explicit per-channel observation masks",
        "Context": "96 rows; observed-history normalization",
        "Horizon": "24 native steps; up to 512 autoregressively",
    }
    code_license: ClassVar[tuple[str, str]] = (
        "Apache-2.0",
        "https://github.com/NVIDIA/structured-data-models/blob/main/LICENSE",
    )
    weights_license: ClassVar[tuple[str, str]] = (
        "Apache-2.0",
        "https://huggingface.co/nvidia/Kumo-Forecast",
    )
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
        pretrained: bool = True,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__(task=Task.regression)
        self.model = MissingnessBackbone(
            device="meta" if pretrained else device
        )
        if pretrained:
            path = download_checkpoint(
                repo_id="nvidia/Kumo-Forecast",
                filename="kumo-forecast-1.2.0/best_model.pt",
                revision="402caa11b7ff585141e912c6743026141f648892",
            )
            ckpt = torch.load(
                path, map_location="cpu", weights_only=True, mmap=True
            )
            self.model.load_state_dict(
                remap_ckpt(ckpt["model_state_dict"]), assign=True
            )
            self.model.to(
                torch.get_default_device() if device is None else device
            )
        self.eval()

    @classmethod
    def default_recipe(cls) -> Recipe:
        """Fit context statistics and average estimator predictions."""
        return default_recipe()

    def forward(
        self,
        x_context: Tensor | TableTensor,
        y_context: Tensor | TableTensor,
        x_query: Tensor | TableTensor,
        *args: Any,
        channel_alignment: Literal["name", "positional"] | None = None,
        history_rows: int | None = None,
        **kwargs: Any,
    ) -> TableTensor:
        """Forecast from ordered, potentially incomplete histories."""
        if channel_alignment is None:
            channel_alignment = (
                "name"
                if isinstance(x_context, TableTensor)
                or isinstance(y_context, TableTensor)
                else "positional"
            )
        if not isinstance(x_context, TableTensor):
            x_context = TableTensor.from_tensor(x_context)
        if not isinstance(y_context, TableTensor):
            y_context = TableTensor.from_tensor(y_context)
        if not isinstance(x_query, TableTensor):
            x_query = TableTensor.from_tensor(x_query)
        if x_query.size(-1):
            raise ValueError(
                "KumoForecastMissingness requires a query with zero columns "
                "(no future covariates)"
            )
        if not 1 <= x_query.size(-2) <= 512:
            raise ValueError(
                "Expected a forecast horizon between 1 and 512 rows"
            )
        if history_rows is not None:
            if history_rows < self.model.context_size:
                raise ValueError("Expected history_rows >= 96")
            x_context = x_context[..., -history_rows:, :]
            y_context = y_context[..., -history_rows:, :]
        if y_context.size(-2) < self.model.context_size:
            raise ValueError("Expected at least 96 context rows")
        if not y_context.numerical[..., -96:, 0].isfinite().any(-1).all():
            raise ValueError(
                "The first target must have observed values "
                "in the last 96 rows"
            )
        return super().forward(
            x_context,
            y_context,
            x_query,
            *args,
            channel_alignment=channel_alignment,
            **kwargs,
        )

    def fit(self, *args: Any, **kwargs: Any) -> None:
        """Use a one-shot forward call for this autoregressive forecaster."""
        raise NotImplementedError(
            "KumoForecastMissingness does not support fit()"
        )

    def _forward(
        self,
        x_context: TableTensor | None,
        y_context: TableTensor | None,
        x_query: TableTensor | None,
        x_context_only: TableTensor | None,
        related_context_tables: RelatedTables[TableTensor] | None,
        related_query_tables: RelatedTables[TableTensor] | None,
        cache: Cache | None,
        generator: torch.Generator | None,
        categorical_mask: Tensor | None = None,
        channel_alignment: Literal["name", "positional"] = "positional",
        **kwargs: Any,
    ) -> TableTensor:
        assert y_context is not None
        assert x_query is not None
        if kwargs:
            raise TypeError(
                f"Unsupported forecasting arguments: {', '.join(kwargs)}"
            )
        columns = list(y_context.numerical_columns)
        x = y_context.numerical
        if x_context_only is not None and x_context_only.size(-1):
            columns.extend(x_context_only.numerical_columns)
            x = torch.cat([x, x_context_only.numerical], -1)
        x = x[..., -self.model.context_size :, :].transpose(-1, -2)
        x = x.to(self.model.lin.weight.dtype)
        if channel_alignment == "positional":
            slots = list(range(len(columns)))
            num_variates = len(columns)
        elif channel_alignment == "name":
            if len(set(columns)) != len(columns) or any(
                name not in self.channel_schema[1:] for name in columns[1:]
            ):
                raise ValueError(
                    "Named auxiliary channels must be unique and belong to "
                    f"{self.channel_schema[1:]}; use "
                    "channel_alignment='positional' for other names"
                )
            slots = [0] + [
                self.channel_schema.index(name) for name in columns[1:]
            ]
            num_variates = len(self.channel_schema)
        else:
            raise ValueError(
                f"Unknown channel alignment: {channel_alignment!r}"
            )
        indices = torch.tensor(slots, device=x.device)
        history = x.new_full(
            (*x.shape[:-2], num_variates, x.size(-1)), torch.nan
        )
        history.index_copy_(-2, indices, x)
        valid = torch.zeros(num_variates, dtype=torch.bool, device=x.device)
        valid[indices] = True
        pieces = []
        horizon = x_query.size(-2)
        for start in range(0, horizon, self.model.prediction_size):
            pred = self.model(history, valid)
            pieces.append(pred.index_select(-2, indices[: y_context.size(-1)]))
            if start + self.model.prediction_size < horizon:
                pred = pred.masked_fill(~valid.unsqueeze(-1), torch.nan)
                history = torch.cat([history, pred], -1)[
                    ..., -self.model.context_size :
                ]
        result = torch.cat(pieces, -1)[..., :horizon].transpose(-1, -2)
        return TableTensor.from_tensor(
            result, columns=y_context.numerical_columns
        )
