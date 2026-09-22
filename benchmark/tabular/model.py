# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SDM model adapters for TabArena and BeyondArena."""

import abc
import copy
import math
from dataclasses import dataclass
from typing import Any, ClassVar, Literal

import numpy as np
import pandas as pd
import torch
from autogluon.core.constants import BINARY, MULTICLASS, REGRESSION
from autogluon.core.models.abstract.shared_weights import SharedWeights
from autogluon.tabular.models.abstract.abstract_torch_model import (
    AbstractTorchModel,
)
from tabarena.benchmark.exec_models import AGModelWrapper

import sdm
import sdm.models.kumo.tabular.model as kumo_tabular
import sdm.processing as sp
from sdm.models.kumo.tabular.model import scale_ecoc_estimators
from sdm.models.kumo.tabular.recipe import default_recipe

Task = Literal["classification", "regression"]


class SDMModel(AbstractTorchModel, abc.ABC):
    """AutoGluon adapter shared by SDM in-context tabular models."""

    ag_priority = 65
    _supported_problem_types: ClassVar[list[str]] = [
        BINARY,
        MULTICLASS,
        REGRESSION,
    ]
    default_num_gpus = 1
    minimum_num_gpus = 1
    default_resources_physical_cores_only = True
    gpu_strongly_recommended = True
    # The harness's untimed warm-up imports these and runs a one-member dummy
    # fit, so imports, the CUDA context and the weights never land in the
    # timed fit.
    warmup_modules: ClassVar[tuple[str, ...]] = ("sdm",)
    cheap_hyperparameters: ClassVar[dict[str, Any]] = {"num_estimators": 1}
    # Bagged fits (the arenas' official protocol) fit one child at a time and
    # refit once on all rows, as the hosted in-context models do.
    _default_ag_args_ensemble_extra: ClassVar[dict[str, Any]] = {
        "fold_fitting_strategy": "sequential_local",
        "refit_folds": True,
    }

    default_num_estimators: ClassVar[int]
    autocast_dtype: ClassVar[torch.dtype]

    @abc.abstractmethod
    def _create_model(
        self,
        task: Task,
        device: torch.device,
    ) -> sdm.models.ICLModel:
        pass

    def _infer_stypes(self, X: pd.DataFrame) -> dict[str, sdm.StypeLike]:
        return sdm.infer_stypes(X)

    def _set_default_params(self) -> None:
        self._set_default_param_value(
            "num_estimators",
            self.default_num_estimators,
        )
        self._set_default_param_value("max_context_size", None)
        self._set_default_param_value("max_columns", None)

    def _recipe(self, params: dict[str, Any]) -> sp.Recipe:
        return self.model.default_recipe()

    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        num_cpus: int = 1,
        num_gpus: int | float = 0,
        **_: Any,
    ) -> None:
        del num_cpus
        self._device = torch.device(
            self._resolve_fit_device(num_gpus=num_gpus)
        )
        task: Task = (
            "regression"
            if self.problem_type == REGRESSION
            else "classification"
        )
        self.model = self._create_model(task=task, device=self._device)

        generator: torch.Generator | None = None
        if self.random_seed is not None:
            generator = torch.Generator(self._device).manual_seed(
                self.random_seed
            )

        X = self.preprocess(X, y=y)
        self.stypes = self._infer_stypes(X)
        x_context = sdm.TableTensor.from_pandas(
            df=X,
            stypes=self.stypes,
            device=self._device,
        )

        target_name = str(y.name) if y.name is not None else "__target__"
        target_stype = (
            "numerical" if self.problem_type == REGRESSION else "categorical"
        )
        y_context = sdm.TableTensor.from_pandas(
            df=y.rename(target_name).to_frame(),
            stypes={target_name: target_stype},
            device=self._device,
        )

        params = self._get_model_params()
        self._num_estimators = params["num_estimators"]
        max_context_size = params["max_context_size"]
        subsamples = max_context_size is not None and len(X) > max_context_size

        recipe = self._recipe(params)
        if params["max_columns"] is not None:
            for processor in recipe.features.modules():
                if isinstance(processor, sp.SelectColumns):
                    processor.max_columns = params["max_columns"]

        num_estimators: int | None = self._num_estimators
        if subsamples:
            num_estimators = scale_ecoc_estimators(
                y=y_context,
                num_estimators=num_estimators,
                recipe=recipe,
            )
            assert num_estimators is not None
            self._num_estimators = num_estimators
            num_repeats = math.ceil(num_estimators * max_context_size / len(X))
            perm = torch.cat(
                [
                    torch.randperm(
                        len(X),
                        generator=generator,
                        device=self._device,
                    )
                    for _ in range(num_repeats)
                ]
            )[: num_estimators * max_context_size]
            shape = (num_estimators, max_context_size)
            x_context = x_context[perm].unflatten(0, shape)
            y_context = y_context[perm].unflatten(0, shape)
            num_estimators = None
        self._expand_query = num_estimators is None

        with torch.amp.autocast(
            self._device.type,
            self.autocast_dtype,
            enabled=x_context.is_cuda,
        ):
            self.model.fit(
                x=x_context,
                y=y_context,
                recipe=recipe,
                num_estimators=num_estimators,
                generator=generator,
            )

    def _predict_proba(
        self,
        X: pd.DataFrame,
        **kwargs: Any,
    ) -> np.ndarray:
        X = self.preprocess(X, **kwargs)
        x_query = sdm.TableTensor.from_pandas(
            df=X,
            stypes=self.stypes,
            device=self._device,
        )
        if self._expand_query:
            x_query = x_query.expand(
                self._num_estimators,
                *x_query.size(),
            )

        with torch.amp.autocast(
            self._device.type,
            self.autocast_dtype,
            enabled=x_query.is_cuda,
        ):
            out = self.model.predict(x_query)

        if self.problem_type == REGRESSION:
            return out.numerical.float().mean(dim=-1).cpu().numpy()

        assert self.num_classes is not None
        columns = out.columns[sdm.Stype.numerical]
        indices = [columns.index(str(i)) for i in range(self.num_classes)]
        probabilities = out.numerical[..., indices].float().cpu().numpy()
        return self._convert_proba_to_unified_form(probabilities)

    def get_device(self) -> str:
        return str(next(self.model.parameters()).device)

    def _set_device(self, device: str) -> None:
        self.model.to(device)
        self._device = torch.device(device)

    def cleanup(self) -> None:
        self.model.clear()
        if self._device.type == "cuda":
            torch.cuda.synchronize(self._device)
            torch._C._host_emptyCache()
            torch.cuda.empty_cache()

    def _more_tags(self) -> dict[str, bool]:
        return {"can_refit_full": True}

    # AutoGluon pickles a model whose network is shared without the weights,
    # but its object walker stops at any torch module, and the served SDM
    # model is one. So the pickle carries a copy of the served model whose
    # networks are placeholders, and the load takes them back from the
    # shared-weights registry.
    def _shared_network(self, task: str) -> torch.nn.Module:
        raise NotImplementedError

    def __getstate__(self) -> dict[str, Any]:
        served = self.__dict__.get("model")
        if served is None or self._shared_state is None:
            return super().__getstate__()
        state = self.__dict__.copy()
        clone = copy.copy(served)
        clone._modules = dict(served._modules)
        clone._modules["models"] = torch.nn.ModuleDict(
            {task: torch.nn.Identity() for task in served.models}
        )
        state["model"] = clone
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        super().__setstate__(state)
        served = self.__dict__.get("model")
        if served is None or self._shared_state is None:
            return
        for task, module in list(served.models.items()):
            if isinstance(module, torch.nn.Identity):
                served.models[task] = self._shared_network(task)


class SDMModelWrapper(AGModelWrapper):
    def cleanup(self) -> None:
        model = getattr(self, "model", None)
        cleanup = getattr(model, "cleanup", None)
        if callable(cleanup):
            cleanup()


class SDMTabICLv2Model(SDMModel):
    ag_key = "SDM-TABICLV2"
    ag_name = "SDMTabICLv2"
    default_num_estimators = 8
    autocast_dtype = torch.float16

    def _create_model(
        self,
        task: Task,
        device: torch.device,
    ) -> sdm.models.TabICLv2:
        return sdm.models.TabICLv2(task=task, device=device)


class SDMKumoTabularModel(SDMModel):
    ag_key = "SDM-KUMO-TABULAR"
    ag_name = "SDMKumoTabular"
    default_num_estimators = 8
    autocast_dtype = torch.float16
    # The network is built once per process and shared by every fit.
    shared_weights: ClassVar[SharedWeights] = SharedWeights(
        loader="sdm.models.kumo.tabular.model:load_network",
        key=("task", "size", "checkpoint"),
    )

    def _set_default_params(self) -> None:
        super()._set_default_params()
        self._set_default_param_value("checkpoint", None)
        self._set_default_param_value("size", "large")
        self._set_default_param_value("numerical_missing", "nan")
        self._set_default_param_value("regression_reduction", "scalar_trim")

    def _recipe(self, params: dict[str, Any]) -> sp.Recipe:
        return default_recipe(
            numerical_missing=params["numerical_missing"],
            regression_reduction=params["regression_reduction"],
        )

    def _create_model(
        self,
        task: Task,
        device: torch.device,
    ) -> sdm.models.KumoTabular:
        params = self._get_model_params()
        checkpoint = params["checkpoint"]
        return sdm.models.KumoTabular(
            task=task,
            size=params["size"],
            device=device,
            checkpoint=None if checkpoint is None else str(checkpoint),
        )

    def _shared_network(self, task: str) -> torch.nn.Module:
        params = self._get_model_params()
        checkpoint = params["checkpoint"]
        # Looked up on the module at call time: the registry wraps it there.
        return kumo_tabular.load_network(
            task=task,
            size=params["size"],
            checkpoint=None if checkpoint is None else str(checkpoint),
            device=self._device,
        )

    def _infer_stypes(self, X: pd.DataFrame) -> dict[str, sdm.StypeLike]:
        # A numeric column of two or three distinct values (missing counted
        # as one) holds category codes once the table is large enough for
        # that to be evidence.
        stypes = sdm.infer_stypes(X)
        if len(X) > 150:
            for column, stype in stypes.items():
                if stype != sdm.Stype.numerical:
                    continue
                if 1 < X[column].nunique(dropna=False) < 4:
                    stypes[column] = sdm.Stype.categorical
        return stypes


class SDMTabFMModel(SDMModel):
    ag_key = "SDM-TABFM"
    ag_name = "SDMTabFM"
    default_num_estimators = 32
    autocast_dtype = torch.bfloat16

    def _create_model(
        self,
        task: Task,
        device: torch.device,
    ) -> sdm.models.TabFM:
        return sdm.models.TabFM(
            task=task,
            accept_license=True,
            device=device,
        )


@dataclass(frozen=True)
class ModelConfig:
    name: str
    model_cls: type[SDMModel]


MODEL_CONFIGS = {
    "tabiclv2": ModelConfig(
        name="TabICLv2",
        model_cls=SDMTabICLv2Model,
    ),
    "kumo-tabular": ModelConfig(
        name="KumoTabular",
        model_cls=SDMKumoTabularModel,
    ),
    "tabfm": ModelConfig(
        name="TabFM",
        model_cls=SDMTabFMModel,
    ),
}
