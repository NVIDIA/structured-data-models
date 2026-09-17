# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SDM model adapters for TabArena and BeyondArena."""

import abc
import math
from collections.abc import Collection
from dataclasses import dataclass
from typing import Any, ClassVar, Literal

import numpy as np
import pandas as pd
import torch
from autogluon.core.constants import BINARY, MULTICLASS, REGRESSION
from autogluon.tabular.models.abstract.abstract_torch_model import (
    AbstractTorchModel,
)
from tabarena.benchmark.exec_models import AGModelWrapper
from tabarena.benchmark.experiment import OOFExperimentRunner

import sdm
import sdm.processing as sp

Task = Literal["classification", "regression"]


def _append_pca(
    recipe: sdm.Recipe,
    num_components: int,
    max_columns: int | None,
    exclude_columns: Collection[str],
    *,
    estimator_share: int = 4,
    center: bool = False,
) -> None:
    """Give principal components to a share of the estimators of ``recipe``.

    Args:
        recipe: Recipe to change in place.
        num_components: Number of principal components to append.
        max_columns: Column limit that the components stay inside, if any.
        exclude_columns: Columns that do not take part in the projection.
        estimator_share: One estimator of this many receives the components.
        center: Subtract the column mean before the projection.
    """
    branch: sp.Processor = sp.PCA(
        num_components,
        append_original=True,
        center=center,
        exclude_columns=exclude_columns,
    )
    if max_columns is not None:
        if num_components >= max_columns:
            raise ValueError(
                f"'pca_components' must stay below the column limit of "
                f"{max_columns}"
            )
        branch = sp.Sequential(
            sp.SelectColumns(max_columns - num_components, method="first"),
            branch,
        )

    # The round robin gives the branch to one estimator of every share.
    plain = [sp.Identity() for _ in range(estimator_share - 1)]
    recipe.append_features(sp.Choice(*plain, branch, method="round_robin"))


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

    default_num_estimators: ClassVar[int]
    autocast_dtype: ClassVar[torch.dtype]

    @staticmethod
    @abc.abstractmethod
    def _create_model(
        task: Task,
        device: torch.device,
    ) -> sdm.models.ICLModel:
        pass

    def _set_default_params(self) -> None:
        self._set_default_param_value(
            "num_estimators",
            self.default_num_estimators,
        )
        self._set_default_param_value("max_context_size", None)
        self._set_default_param_value("max_columns", None)
        self._set_default_param_value("pca_components", 8)

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
        self.stypes = sdm.infer_stypes(X)
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
        num_estimators: int | None = self._num_estimators
        if max_context_size is not None and len(X) > max_context_size:
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

        recipe = self.model.default_recipe()
        if params["max_columns"] is not None:
            for processor in recipe.features.modules():
                if isinstance(processor, sp.SelectColumns):
                    processor.max_columns = params["max_columns"]

        if params["pca_components"]:
            column_limits = [
                processor.max_columns
                for processor in recipe.features.modules()
                if isinstance(processor, sp.SelectColumns)
            ]
            _append_pca(
                recipe,
                params["pca_components"],
                min(column_limits, default=None),
                x_context.columns[sdm.Stype.categorical],
            )

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


class SDMModelWrapper(AGModelWrapper):
    def cleanup(self) -> None:
        model = getattr(self, "model", None)
        cleanup = getattr(model, "cleanup", None)
        if callable(cleanup):
            cleanup()


class SDMExperimentRunner(OOFExperimentRunner):
    def run(self) -> dict:
        try:
            return self._run()
        finally:
            if self.cleanup and getattr(self, "model", None) is not None:
                self._cleanup()


class SDMTabICLv2Model(SDMModel):
    ag_key = "SDM-TABICLV2"
    ag_name = "SDMTabICLv2"
    default_num_estimators = 8
    autocast_dtype = torch.float16

    @staticmethod
    def _create_model(
        task: Task,
        device: torch.device,
    ) -> sdm.models.TabICLv2:
        return sdm.models.TabICLv2(task=task, device=device)


class SDMKumoTabularModel(SDMModel):
    ag_key = "SDM-KUMO-TABULAR"
    ag_name = "SDMKumoTabular"
    default_num_estimators = 8
    autocast_dtype = torch.float16

    @staticmethod
    def _create_model(
        task: Task,
        device: torch.device,
    ) -> sdm.models.KumoTabular:
        return sdm.models.KumoTabular(task=task, device=device)


class SDMTabFMModel(SDMModel):
    ag_key = "SDM-TABFM"
    ag_name = "SDMTabFM"
    default_num_estimators = 32
    autocast_dtype = torch.bfloat16

    @staticmethod
    def _create_model(
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

    @property
    def tabarena_method_name(self) -> str:
        return f"{self.model_cls.ag_name}_c1_default"

    @property
    def beyondarena_method_name(self) -> str:
        return f"{self.model_cls.ag_name}_c1"


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
