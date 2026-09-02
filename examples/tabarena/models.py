"""SDM tabular model adapter for TabArena."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal, Self

import pandas as pd
import torch
from autogluon.core.data import LabelCleaner
from tabarena.benchmark.exec_models.external import ExternalSystemModel

import sdm

Task = Literal["classification", "regression"]
ModelFactory = Callable[[Task, torch.device], sdm.models.ICLModel]


@dataclass(frozen=True)
class ModelConfig:
    name: str
    factory: ModelFactory
    num_estimators: int
    autocast_dtype: torch.dtype

    @property
    def system_name(self) -> str:
        return f"SDM{self.name}System"

    @property
    def method_name(self) -> str:
        return f"{self.system_name}_c1_default"


@lru_cache(maxsize=2)
def _create_tabiclv2(
    task: Task,
    device: torch.device,
) -> sdm.models.TabICLv2:
    return sdm.models.TabICLv2(task=task, device=device)


@lru_cache(maxsize=2)
def _create_kumo_tabular(
    task: Task,
    device: torch.device,
) -> sdm.models.KumoTabular:
    return sdm.models.KumoTabular(task=task, device=device)


@lru_cache(maxsize=1)
def _create_tabfm(
    task: Task,
    device: torch.device,
) -> sdm.models.TabFM:
    return sdm.models.TabFM(task=task, accept_license=True, device=device)


MODEL_CONFIGS = {
    "tabiclv2": ModelConfig(
        name="TabICLv2",
        factory=_create_tabiclv2,
        num_estimators=8,
        autocast_dtype=torch.float16,
    ),
    "kumo-tabular": ModelConfig(
        name="KumoTabular",
        factory=_create_kumo_tabular,
        num_estimators=8,
        autocast_dtype=torch.float16,
    ),
    "tabfm": ModelConfig(
        name="TabFM",
        factory=_create_tabfm,
        num_estimators=8,
        autocast_dtype=torch.bfloat16,
    ),
}


class SDMSystem(ExternalSystemModel):
    def __init__(
        self,
        *,
        model: str,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._config = MODEL_CONFIGS[model]

    def _fit_system(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        target_name: str,
        problem_type: str,
        random_state: int | None,
        **_: object,
    ) -> Self:
        self._device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        task: Task = (
            "regression" if problem_type == "regression" else "classification"
        )
        self.model = self._config.factory(task, self._device)
        generator = None
        if random_state is not None:
            generator = torch.Generator(device=self._device).manual_seed(
                random_state
            )

        self.stypes = sdm.infer_stypes(X)
        target_name = target_name or "__target__"
        if problem_type == "regression":
            target_stype = "numerical"
        else:
            target_stype = "categorical"
            cleaner = LabelCleaner.construct(problem_type=problem_type, y=y)
            self._class_labels_by_key = {
                str(label): label for label in cleaner.ordered_class_labels
            }

        table_x = sdm.TableTensor.from_pandas(
            df=X,
            stypes=self.stypes,
            device=self._device,
        )
        table_y = sdm.TableTensor.from_pandas(
            df=y.rename(target_name).to_frame(),
            stypes={target_name: target_stype},
            device=self._device,
        )
        with torch.amp.autocast(
            self._device.type,
            self._config.autocast_dtype,
            enabled=table_x.is_cuda,
        ):
            self.model.fit(
                x=table_x,
                y=table_y,
                num_estimators=self._config.num_estimators,
                generator=generator,
            )
        return self

    def _predict(self, X: pd.DataFrame) -> pd.Series:
        table_x = sdm.TableTensor.from_pandas(
            df=X,
            stypes=self.stypes,
            device=self._device,
        )
        with torch.amp.autocast(
            self._device.type,
            self._config.autocast_dtype,
            enabled=table_x.is_cuda,
        ):
            out = self.model.predict(table_x)
        values = out.numerical.float().mean(dim=-1).cpu().numpy()
        return pd.Series(values, index=X.index)

    def _predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        table_x = sdm.TableTensor.from_pandas(
            df=X,
            stypes=self.stypes,
            device=self._device,
        )
        with torch.amp.autocast(
            self._device.type,
            self._config.autocast_dtype,
            enabled=table_x.is_cuda,
        ):
            probabilities = self.model.predict(table_x).to_pandas()
        probabilities.index = X.index
        probabilities = probabilities.rename(
            columns=self._class_labels_by_key,
        )
        return probabilities.reindex(
            columns=tuple(self._class_labels_by_key.values()),
        )
