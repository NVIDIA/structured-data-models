# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SDM model adapters for TabArena and BeyondArena."""

import abc
import copy
import math
from dataclasses import dataclass
from typing import Any, ClassVar, Literal, cast

import numpy as np
import pandas as pd
import torch
from autogluon.core.constants import BINARY, MULTICLASS, REGRESSION
from autogluon.core.models.abstract.shared_weights import SharedWeights
from autogluon.tabular.models.abstract.abstract_torch_model import (
    AbstractTorchModel,
)
from tabarena.models.warmup import warmup_torch

import sdm
import sdm.processing as sp
from sdm.processing.execution import RecipeExecution

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

    default_num_estimators: ClassVar[int]
    autocast_dtype: ClassVar[torch.dtype]

    @staticmethod
    @abc.abstractmethod
    def _create_model(
        task: Task,
        device: torch.device,
    ) -> sdm.models.ICLModel:
        pass

    def _create_recipe(self) -> sdm.Recipe:
        return self.model.default_recipe()

    def _set_default_params(self) -> None:
        self._set_default_param_value(
            "num_estimators",
            self.default_num_estimators,
        )
        self._set_default_param_value("max_context_size", None)
        self._set_default_param_value("max_columns", None)
        self._set_default_param_value("kv_cache", False)

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

        recipe = self._create_recipe()
        if params["max_columns"] is not None:
            for processor in recipe.features.modules():
                if isinstance(processor, sp.SelectColumns):
                    processor.max_columns = params["max_columns"]

        self._recipe_execution: RecipeExecution | None = None
        if params["kv_cache"]:
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
            return

        self._recipe_execution = RecipeExecution(recipe)
        # KumoTabular and TabFM need the column types before preprocessing.
        self._schema = x_context.schema
        with (
            torch.inference_mode(),
            torch.amp.autocast(self._device.type, enabled=False),
        ):
            contexts = self._recipe_execution.fit_transform(
                x=x_context,
                y=y_context,
                related_tables=None,
                num_members=num_estimators,
                generator=generator,
            )
        self._contexts = tuple(
            context._replace(
                x=cast(sdm.TableTensor, context.x.cpu()),
                y=cast(sdm.TableTensor, context.y.cpu()),
            )
            for context in contexts
        )
        # Replay the same model-side randomness as the cached fit path.
        if generator is not None:
            self._rng_state = generator.get_state()
        elif self._device.type == "cuda":
            self._rng_state = torch.cuda.get_rng_state(self._device)
        else:
            self._rng_state = torch.get_rng_state()

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

        if self._recipe_execution is None:
            with torch.amp.autocast(
                self._device.type,
                self.autocast_dtype,
                enabled=x_query.is_cuda,
            ):
                out = self.model.predict(x_query)
        else:
            with (
                torch.inference_mode(),
                torch.amp.autocast(self._device.type, enabled=False),
            ):
                queries = self._recipe_execution.transform(
                    x=x_query,
                    related_tables=None,
                )
                generator = torch.Generator(self._device).set_state(
                    self._rng_state
                )
                outputs = []
                for context, query in zip(self._contexts, queries):
                    with torch.amp.autocast(
                        self._device.type,
                        self.autocast_dtype,
                        enabled=x_query.is_cuda,
                    ):
                        out = self.model._forward(
                            x_context=context.x.to(self._device),
                            y_context=context.y.to(self._device),
                            x_query=query.x,
                            related_context_tables=None,
                            related_query_tables=None,
                            cache=None,
                            generator=generator,
                            _schema=self._schema,
                        )
                    outputs.append(out.to(query.x.dtype))

                if self.problem_type == REGRESSION:
                    outputs = list(
                        self._recipe_execution.inverse_transform_target(
                            outputs
                        )
                    )
                out = self._recipe_execution.transform_output(outputs)

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
        if self._recipe_execution is not None:
            recipe = self._recipe_execution.recipe
            for processor in (recipe.features, recipe.target, recipe.output):
                processor.to(device)
        self._device = torch.device(device)

    def _more_tags(self) -> dict[str, bool]:
        return {"can_refit_full": True}


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


def _load_kumo_network(*, task: str, device: torch.device) -> torch.nn.Module:
    return sdm.models.KumoTabular(task=task, device=device).models[task]


class SDMKumoTabularModel(SDMModel):
    ag_key = "SDM-KUMO-TABULAR"
    ag_name = "SDMKumoTabular"
    default_num_estimators = 8
    autocast_dtype = torch.float16
    # Bagged children are fit one at a time in this process, so they share the
    # pretrained network of their task through AutoGluon's registry.
    _default_ag_args_ensemble_extra: ClassVar[dict[str, Any]] = {
        "fold_fitting_strategy": "sequential_local",
    }
    shared_weights: ClassVar[SharedWeights] = SharedWeights(
        loader="benchmark.tabular.model:_load_kumo_network",
        key=("task",),
    )

    @classmethod
    def warmup(
        cls,
        *,
        problem_type: str | None = None,
        num_cpus: int | None = None,
        num_gpus: float | None = None,
        hyperparameters: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        warmup_torch(cuda=None if num_gpus is None else num_gpus > 0)

    @staticmethod
    def _create_model(
        task: Task,
        device: torch.device,
    ) -> sdm.models.KumoTabular:
        model = sdm.models.KumoTabular(
            task=task,
            pretrained=False,
            device="meta",
        )
        model.models[task] = _load_kumo_network(task=task, device=device)
        return model

    # AutoGluon does not look inside the served model for the shared network,
    # so the pickle holds a placeholder and the load restores the network on
    # the fit device.
    def __getstate__(self) -> dict[str, Any]:
        if self.model is None or self._shared_state is None:
            return super().__getstate__()
        served = cast(sdm.models.KumoTabular, self.model)
        model = copy.copy(served)
        model._modules = dict(served._modules)
        model._modules["models"] = torch.nn.ModuleDict(
            modules={task: torch.nn.Identity() for task in served.models},
        )
        return {**self.__dict__, "model": model}

    def __setstate__(self, state: dict[str, Any]) -> None:
        super().__setstate__(state)
        if self.model is None or self._shared_state is None:
            return
        served = cast(sdm.models.KumoTabular, self.model)
        for task in served.models:
            served.models[task] = _load_kumo_network(
                task=task,
                device=self._device,
            )

    def _create_recipe(self) -> sdm.Recipe:
        recipe = super()._create_recipe()
        # TabArena aligns features with its fitted generator and targets with
        # its label cleaner.
        for pipeline in (recipe.target,):
            stype_dispatch = next(
                processor
                for processor in pipeline.modules()
                if isinstance(processor, sp.StypeDispatch)
            )
            categorical = stype_dispatch.processors[str(sdm.Stype.categorical)]
            assert isinstance(categorical, sp.Sequential)
            processors = iter(categorical)
            assert isinstance(next(processors), sp.AlignCategories)
            stype_dispatch.processors[str(sdm.Stype.categorical)] = (
                sp.Sequential(*processors)
            )
        return recipe


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
