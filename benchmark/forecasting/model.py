# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SDM model adapter shared by the time-series forecasting benchmarks."""

import argparse
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

import sdm
from sdm.models.timesfm3 import TimesFM3


@dataclass(frozen=True)
class ModelConfig:
    name: str
    factory: Callable[[torch.device], sdm.models.ICLModel]
    num_estimators: int = 1
    max_context_size: int | None = None


MODEL_CONFIGS = {
    "timesfm3": ModelConfig(
        name="TimesFM3",
        factory=lambda device: TimesFM3(accept_license=True, device=device),
        num_estimators=2,
        max_context_size=15360,
    ),
}


def add_model_args(parser: argparse.ArgumentParser) -> None:
    """Add the model flags shared by all forecasting benchmarks."""
    parser.add_argument(
        "--model",
        choices=tuple(MODEL_CONFIGS),
        default="timesfm3",
        help="SDM model to benchmark.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Number of series forecast per model call.",
    )
    parser.add_argument(
        "--max_context_size",
        type=int,
        help="Truncate each context to its most recent time steps "
        "(defaults to the model's maximum context size).",
    )


def forecast(
    model: sdm.models.ICLModel,
    config: ModelConfig,
    past: Sequence[pd.DataFrame],  # B x [T_i, V + C]
    future: Sequence[pd.DataFrame],  # B x [H, K]
    targets: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:  # [B, H, V, Q], [Q]
    """Forecast the quantiles of a batch of (multivariate) time series.

    Time steps map to rows: ``past`` holds the targets and all covariates of
    the context, while ``future`` holds the covariates known over the
    forecast horizon. Columns keep their semantic types, so that the model
    decides which ones to consume. Missing values are passed through.
    """
    device = next(model.parameters()).device
    num_series, horizon = len(past), len(future[0])
    size = max(len(df) for df in past)
    if config.max_context_size is not None:
        size = min(size, config.max_context_size)

    # Left-pad shorter series with missing values so that all series end at
    # the forecast start:
    past = [df.iloc[-size:].reset_index(drop=True) for df in past]
    past = [df.reindex(range(len(df) - size, len(df))) for df in past]

    # Infer semantic types jointly so that context and query share them, and
    # order targets first so that forecasts follow ``targets``:
    df = pd.concat([*past, *future], ignore_index=True)
    df = df[[*targets, *df.columns.drop(targets)]]
    table = sdm.TableTensor.from_pandas(
        df=df,
        stypes=sdm.infer_stypes(df, id="infer"),
        device=device,
    )
    # Split rows into the context [B, T, V + C] and the query [B, H, V + C]:
    context = table[: num_series * size].view(num_series, size, -1)
    query = table[num_series * size :].view(num_series, horizon, -1)
    assert isinstance(context, sdm.TableTensor)
    assert isinstance(query, sdm.TableTensor)

    out = model(
        x_context=context.drop_columns(targets),
        y_context=context.select_columns(targets),
        x_query=query.select_columns(future[0].columns),
        num_estimators=config.num_estimators,
    )
    values, levels = sdm.evaluation.to_quantiles(out)
    return values.cpu().numpy(), levels.cpu().numpy()


def select_levels(
    values: np.ndarray,  # [..., Q]
    levels: np.ndarray,  # [Q]
    wanted: Sequence[float],
) -> np.ndarray:  # [..., len(wanted)]
    """Select the quantile levels requested by a benchmark."""
    index = [int(np.abs(levels - level).argmin()) for level in wanted]
    if not np.allclose(levels[index], wanted):
        raise ValueError(
            f"Model does not predict quantile levels {list(wanted)} "
            f"(got {levels.tolist()})"
        )
    return values[..., index]
