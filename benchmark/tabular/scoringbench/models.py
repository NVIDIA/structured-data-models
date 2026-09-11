"""SDM probabilistic-regression adapters for ScoringBench."""

from __future__ import annotations

import abc
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import pandas as pd
import torch
from scoringbench.univariate.wrappers import (
    DistributionPrediction,
    ProbabilisticWrapper,
    quantiles_to_distribution,
)

import sdm

ModelFactory = Callable[[torch.device], sdm.models.ICLModel]

# ScoringBench's reference TabICL adapter evaluates these exact levels. SDM
# exposes the model's 999 native thousandth levels, which are interpolated to
# match the reference adapter.
QUANTILE_LEVELS = np.linspace(0.005, 0.995, 200)
MODEL_QUANTILE_COLUMNS = tuple(f"q{index:03d}" for index in range(1, 1000))


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for an SDM ScoringBench model."""

    name: str
    method: str
    factory: ModelFactory
    autocast_dtype: torch.dtype
    num_estimators: int


def _create_tabiclv2(device: torch.device) -> sdm.models.TabICLv2:
    return sdm.models.TabICLv2(task="regression", device=device)


def _create_kumo_tabular(device: torch.device) -> sdm.models.KumoTabular:
    return sdm.models.KumoTabular(task="regression", device=device)


MODEL_CONFIGS = {
    "tabiclv2": ModelConfig(
        name="TabICLv2",
        method="sdm_tabiclv2",
        factory=_create_tabiclv2,
        autocast_dtype=torch.float16,
        num_estimators=8,
    ),
    "kumo-tabular": ModelConfig(
        name="KumoTabular",
        method="sdm_kumo_tabular",
        factory=_create_kumo_tabular,
        autocast_dtype=torch.float16,
        num_estimators=8,
    ),
}


class SDMQuantileWrapper(ProbabilisticWrapper, abc.ABC):
    """Run an SDM quantile model through ScoringBench's wrapper contract."""

    config: ClassVar[ModelConfig]

    def __init__(
        self,
        *,
        device: torch.device | str | None = None,
        seed: int = 42,
        batch_size: int | None = None,
    ) -> None:
        self.device = torch.device(
            "cuda"
            if device is None and torch.cuda.is_available()
            else device or "cpu"
        )
        self.seed = seed
        self.batch_size = batch_size
        # ScoringBench constructs wrappers outside its fit timer. Constructing
        # the model here matches the timing boundary of its native wrappers.
        self.model = self.config.factory(self.device)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> SDMQuantileWrapper:
        self._set_train_range(y)
        self.stypes = sdm.infer_stypes(X)
        self.x_context = sdm.TableTensor.from_pandas(
            df=X,
            stypes=self.stypes,
            device=self.device,
        )
        target_name = str(y.name) if y.name is not None else "__target__"
        self.y_context = sdm.TableTensor.from_pandas(
            df=y.rename(target_name).to_frame(),
            stypes={target_name: "numerical"},
            device=self.device,
        )
        return self

    def _batches(self, X: pd.DataFrame) -> Iterator[pd.DataFrame]:
        if self.batch_size is None or len(X) <= self.batch_size:
            yield X
            return
        for start in range(0, len(X), self.batch_size):
            yield X.iloc[start : start + self.batch_size]

    def _predict_quantiles(self, X: pd.DataFrame) -> np.ndarray:
        chunks = []
        for batch in self._batches(X):
            x_query = sdm.TableTensor.from_pandas(
                df=batch,
                stypes=self.stypes,
                device=self.device,
            )
            with torch.amp.autocast(
                self.device.type,
                self.config.autocast_dtype,
                enabled=x_query.is_cuda,
            ):
                out = self.model(
                    x_context=self.x_context,
                    y_context=self.y_context,
                    x_query=x_query,
                    num_estimators=self.config.num_estimators,
                    generator=torch.Generator(device=self.device).manual_seed(
                        self.seed
                    ),
                )
            columns = out.columns[sdm.Stype.numerical]
            positions = [
                columns.index(column) for column in MODEL_QUANTILE_COLUMNS
            ]
            quantiles = out.numerical[..., positions]
            model_levels = torch.linspace(
                0.0,
                1.0,
                len(MODEL_QUANTILE_COLUMNS) + 2,
                device=quantiles.device,
                dtype=quantiles.dtype,
            )[1:-1]
            levels = quantiles.new_tensor(QUANTILE_LEVELS)
            lower = (
                torch.searchsorted(
                    model_levels[:-1].contiguous(),
                    levels.contiguous(),
                    right=True,
                )
                - 1
            ).clamp(0, len(MODEL_QUANTILE_COLUMNS) - 2)
            upper = lower + 1
            weight = (levels - model_levels[lower]) / (
                model_levels[upper] - model_levels[lower]
            )
            interpolated = quantiles[..., lower] + weight * (
                quantiles[..., upper] - quantiles[..., lower]
            )
            chunks.append(interpolated.float().cpu().numpy())
        return np.concatenate(chunks)

    def predict_distribution(
        self,
        X: pd.DataFrame,
    ) -> DistributionPrediction:
        assert self._y_train_range is not None
        return quantiles_to_distribution(
            self._predict_quantiles(X),
            QUANTILE_LEVELS,
            train_range=self._y_train_range,
        )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.predict_distribution(X).mean


class SDMTabICLv2Wrapper(SDMQuantileWrapper):
    config = MODEL_CONFIGS["tabiclv2"]


class SDMKumoTabularWrapper(SDMQuantileWrapper):
    config = MODEL_CONFIGS["kumo-tabular"]


WRAPPERS: dict[str, type[SDMQuantileWrapper]] = {
    "tabiclv2": SDMTabICLv2Wrapper,
    "kumo-tabular": SDMKumoTabularWrapper,
}
