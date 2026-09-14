"""SDM probabilistic-regression adapters for ScoringBench."""

from __future__ import annotations

import abc
from collections.abc import Callable
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


@dataclass(frozen=True)
class ModelConfig:
    name: str
    method: str
    factory: Callable[[torch.device], sdm.models.ICLModel]
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
    config: ClassVar[ModelConfig]

    def __init__(
        self,
        *,
        device: torch.device | str | None = None,
        seed: int = 42,
        batch_size: int | None = None,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)
        self.seed = seed
        self.batch_size = batch_size
        self.model = self.config.factory(self.device)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> SDMQuantileWrapper:
        self._set_train_range(y)
        self.stypes = sdm.infer_stypes(X)

        x_context = sdm.TableTensor.from_pandas(
            df=X,
            stypes=self.stypes,
            device=self.device,
        )
        target_name = str(y.name) if y.name is not None else "__target__"
        y_context = sdm.TableTensor.from_pandas(
            df=y.rename(target_name).to_frame(),
            stypes={target_name: "numerical"},
            device=self.device,
        )

        generator = torch.Generator(device=self.device).manual_seed(self.seed)
        with torch.amp.autocast(
            self.device.type,
            self.config.autocast_dtype,
            enabled=x_context.is_cuda,
        ):
            self.model.fit(
                x=x_context,
                y=y_context,
                num_estimators=self.config.num_estimators,
                generator=generator,
            )

        return self

    def predict_distribution(self, X: pd.DataFrame) -> DistributionPrediction:
        x_query = sdm.TableTensor.from_pandas(
            df=X,
            stypes=self.stypes,
            device=self.device,
        )

        outs = []
        for batch in x_query.split(self.batch_size or len(x_query), dim=-2):
            with torch.amp.autocast(
                self.device.type,
                self.config.autocast_dtype,
                enabled=x_query.is_cuda,
            ):
                outs.append(self.model.predict(x=batch).numerical)

        assert self._y_train_range is not None
        return quantiles_to_distribution(
            torch.cat(outs, dim=-2).cpu().numpy(),
            np.linspace(0.001, 0.999, 999),
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
