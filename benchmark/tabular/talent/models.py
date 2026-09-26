# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SDM tabular model adapter for TALENT."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import lru_cache, partial
from typing import Any, Literal, cast

import numpy as np
import torch
import torch.nn.functional as F
from TALENT.model.lib.data import (
    Dataset,
    data_label_process,
    data_nan_process,
)
from TALENT.model.method_registry import (
    METHOD_REGISTRY,
    Architecture,
    Hardware,
    MethodSpec,
    OutputType,
)
from TALENT.model.methods.base import Method

import sdm

Task = Literal["classification", "regression"]
ModelFactory = Callable[[Task, torch.device], sdm.models.ICLModel]


class UnsupportedDatasetError(RuntimeError):
    """The selected model cannot represent a dataset's target."""


@dataclass(frozen=True)
class ModelConfig:
    name: str
    factory: ModelFactory
    num_estimators: int
    autocast_dtype: torch.dtype
    max_classes: int | None = None


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
    size: Literal["small", "medium", "large"],
) -> sdm.models.KumoTabular:
    return sdm.models.KumoTabular(task=task, size=size, device=device)


@lru_cache(maxsize=1)
def _create_tabfm(
    task: Task,
    device: torch.device,
) -> sdm.models.TabFM:
    return sdm.models.TabFM(
        task=task,
        accept_license=True,
        device=device,
    )


MODEL_CONFIGS = {
    "tabiclv2": ModelConfig(
        name="TabICLv2",
        factory=_create_tabiclv2,
        num_estimators=8,
        autocast_dtype=torch.float16,
    ),
    "kumo-tabular-small": ModelConfig(
        name="KumoTabular-Small",
        factory=partial(_create_kumo_tabular, size="small"),
        num_estimators=8,
        autocast_dtype=torch.float16,
    ),
    "kumo-tabular-medium": ModelConfig(
        name="KumoTabular-Medium",
        factory=partial(_create_kumo_tabular, size="medium"),
        num_estimators=8,
        autocast_dtype=torch.float16,
    ),
    "kumo-tabular-large": ModelConfig(
        name="KumoTabular-Large",
        factory=partial(_create_kumo_tabular, size="large"),
        num_estimators=16,
        autocast_dtype=torch.float16,
    ),
    "tabfm": ModelConfig(
        name="TabFM",
        factory=_create_tabfm,
        num_estimators=8,
        autocast_dtype=torch.bfloat16,
        max_classes=10,
    ),
}


def _target_vector(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim == 1:
        return values
    if values.ndim == 2 and values.shape[1] == 1:
        return values[:, 0]
    raise ValueError(f"Expected one target column, got shape {values.shape}.")


class SDMMethod(Method):
    """Run an SDM in-context model through TALENT's evaluation protocol."""

    def __init__(self, args: Any, is_regression: bool) -> None:
        super().__init__(args, is_regression)
        assert args.normalization == "none"
        assert args.cat_policy == "indices"
        assert args.num_policy == "none"
        assert args.tune is not True

        general = args.config.get("general", {}) or {}
        self._config = MODEL_CONFIGS[general["model"]]
        self._device = torch.device(general.get("device", args.device))
        self.args.device = self._device
        self._num_estimators = general.get(
            "num_estimators",
            self._config.num_estimators,
        )

    def data_format(
        self,
        is_train: bool = True,
        N: dict[str, np.ndarray] | None = None,
        C: dict[str, np.ndarray] | None = None,
        y: dict[str, np.ndarray] | None = None,
    ) -> None:
        if is_train:
            (
                self.N,
                self.C,
                self.num_new_value,
                self.imputer,
                self.cat_new_value,
            ) = data_nan_process(
                self.N,
                self.C,
                self.args.num_nan_policy,
                self.args.cat_nan_policy,
            )
            self.y, self.y_info, self.label_encoder = data_label_process(
                self.y,
                self.is_regression,
            )
            self.y = {
                split: _target_vector(values)
                for split, values in self.y.items()
            }
            self.criterion = F.mse_loss if self.is_regression else F.nll_loss
            return

        N_test, C_test, _, _, _ = data_nan_process(
            N,
            C,
            self.args.num_nan_policy,
            self.args.cat_nan_policy,
            self.num_new_value,
            self.imputer,
            self.cat_new_value,
        )
        assert y is not None
        y_test, _, _ = data_label_process(
            y,
            self.is_regression,
            self.y_info,
            self.label_encoder,
        )
        self.N_test = None if N_test is None else N_test["test"]
        self.C_test = None if C_test is None else C_test["test"]
        self.y_test = _target_vector(y_test["test"])

    def _to_table(
        self,
        numerical: np.ndarray | None,
        categorical: np.ndarray | None,
    ) -> sdm.TableTensor:
        data: dict[str, Sequence[Any]] = {}
        stypes: dict[str, str] = {}
        if numerical is not None:
            numerical = np.asarray(numerical)
            for i, column in enumerate(numerical.T):
                name = f"num_{i}"
                data[name] = cast(Sequence[Any], column)
                stypes[name] = "numerical"
        if categorical is not None:
            categorical = np.asarray(categorical)
            for i, column in enumerate(categorical.T):
                name = f"cat_{i}"
                data[name] = cast(Sequence[Any], column)
                stypes[name] = "categorical"
        return sdm.TableTensor.from_columns(
            data=data,
            stypes=stypes,
            device=self._device,
        )

    def _to_target(self, values: np.ndarray) -> sdm.TableTensor:
        return sdm.TableTensor.from_columns(
            data={"target": cast(Sequence[Any], values)},
            stypes={
                "target": (
                    "numerical" if self.is_regression else "categorical"
                )
            },
            device=self._device,
        )

    def fit(
        self,
        data: tuple[Any, Any, Any],
        info: dict[str, Any],
        train: bool = True,
        config: dict[str, Any] | None = None,
    ) -> None:
        del train, config
        N, C, y = data
        self.D = Dataset(N, C, y, info)
        self.N, self.C, self.y = self.D.N, self.D.C, self.D.y
        self.is_binclass, self.is_multiclass, self.is_regression = (
            self.D.is_binclass,
            self.D.is_multiclass,
            self.D.is_regression,
        )
        self.data_format(is_train=True)

        num_classes = None if self.is_regression else self.y_info["n_classes"]
        if (
            num_classes is not None
            and self._config.max_classes is not None
            and num_classes > self._config.max_classes
        ):
            raise UnsupportedDatasetError(
                f"{self._config.name} supports at most "
                f"{self._config.max_classes} classes, got {num_classes}."
            )

        N_train = None if self.N is None else self.N["train"]
        C_train = None if self.C is None else self.C["train"]
        x_train = self._to_table(N_train, C_train)
        y_train = self._to_target(self.y["train"])
        task: Task = "regression" if self.is_regression else "classification"
        self.model = self._config.factory(task, self._device)
        generator = torch.Generator(device=self._device).manual_seed(
            self.args.seed
        )

        tic = time.perf_counter()
        with torch.amp.autocast(
            self._device.type,
            self._config.autocast_dtype,
            enabled=x_train.is_cuda,
        ):
            self.model.fit(
                x=x_train,
                y=y_train,
                num_estimators=self._num_estimators,
                generator=generator,
            )
        if x_train.is_cuda:
            torch.cuda.synchronize(x_train.device)
        self.fit_time = time.perf_counter() - tic

    def predict(
        self,
        data: tuple[Any, Any, Any],
        info: dict[str, Any],
        model_name: str,
    ) -> tuple[float, tuple[float, ...], tuple[str, ...], np.ndarray]:
        del model_name
        N, C, y = data
        self.data_format(False, N, C, y)
        x_test = self._to_table(self.N_test, self.C_test)

        tic = time.perf_counter()
        with torch.amp.autocast(
            self._device.type,
            self._config.autocast_dtype,
            enabled=x_test.is_cuda,
        ):
            out = self.model.predict(x_test)
        if x_test.is_cuda:
            torch.cuda.synchronize(x_test.device)
        self.predict_time = time.perf_counter() - tic

        if self.is_regression:
            prediction = out.numerical.float().mean(dim=-1).cpu().numpy()
            pred_tensor = torch.as_tensor(prediction).reshape(-1)
            label_tensor = torch.as_tensor(
                self.y_test,
                dtype=torch.float32,
            ).reshape(-1)
            loss = self.criterion(pred_tensor, label_tensor).item()
        else:
            columns = [str(value) for value in self.y_info["classes"]]
            prediction = out.to_pandas()[columns].to_numpy()
            probabilities = torch.as_tensor(prediction)
            loss = self.criterion(
                probabilities.clamp_min(
                    torch.finfo(probabilities.dtype).tiny
                ).log(),
                torch.as_tensor(self.y_test, dtype=torch.long),
            ).item()

        metrics, metric_names = self.metric(
            prediction,
            self.y_test,
            self.y_info,
        )
        if self.is_regression and self.y_info.get("policy") == "mean_std":
            prediction = prediction * self.y_info["std"] + self.y_info["mean"]
        return loss, metrics, metric_names, prediction


def register_sdm_method() -> None:
    """Register the example adapter with TALENT's public method registry."""
    spec = MethodSpec(
        name="sdm",
        module=SDMMethod.__module__,
        class_name=SDMMethod.__name__,
        architecture=Architecture.DEEP,
        hardware=Hardware.GPU,
        output_type=OutputType.PROBABILITIES,
        cat_policy=("indices",),
        normalization="none",
        num_policy="none",
        supports_hpo=False,
        supports_regression=True,
        supports_classification=True,
        train_row_limit=None,
        notes="structured-data-models TALENT benchmark adapter.",
    )
    existing = METHOD_REGISTRY.get(spec.name)
    if existing is not None and existing != spec:
        raise RuntimeError(
            "TALENT already has a different method registered as 'sdm'."
        )
    METHOD_REGISTRY[spec.name] = spec
