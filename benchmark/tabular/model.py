# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SDM model adapters for TabArena and BeyondArena."""

import abc
import math
from dataclasses import dataclass
from typing import Any, ClassVar, Literal, get_args

import numpy as np
import pandas as pd
import torch
from autogluon.core.constants import BINARY, MULTICLASS, REGRESSION
from autogluon.core.models.greedy_ensemble.ensemble_selection import (
    EnsembleSelection,
)
from autogluon.tabular.models.abstract.abstract_torch_model import (
    AbstractTorchModel,
)
from sklearn.model_selection import KFold, StratifiedKFold
from tabarena.benchmark.exec_models import AGModelWrapper
from tabarena.benchmark.experiment import OOFExperimentRunner

import sdm
import sdm.processing as sp
from sdm.models.kumo.tabular.model import scale_ecoc_estimators
from sdm.models.kumo.tabular.recipe import Normalize, default_recipe

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
        self._set_default_param_value("recipe_ensemble", None)
        self._set_default_param_value("ensemble_folds", 3)
        self._set_default_param_value("ensemble_iterations", 20)

    def _recipe(self, params: dict[str, Any]) -> sp.Recipe:
        return self.model.default_recipe()

    def _recipes(self, params: dict[str, Any]) -> dict[str, sp.Recipe]:
        """Named recipes; several are blended with context-fitted weights."""
        return {"default": self._recipe(params)}

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

        generator = self._generator()

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

        recipes = self._recipes(params)
        if subsamples and len(recipes) > 1:
            raise ValueError(
                "recipe_ensemble fits weights on the full context; drop "
                "max_context_size"
            )
        if params["max_columns"] is not None:
            for recipe in recipes.values():
                for processor in recipe.features.modules():
                    if isinstance(processor, sp.SelectColumns):
                        processor.max_columns = params["max_columns"]

        num_estimators: int | None = self._num_estimators
        if subsamples:
            (recipe,) = recipes.values()
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

        self._context: tuple[torch.Tensor, torch.Tensor] | None = None
        self._members: list[tuple[int, sp.Recipe, float]] = []
        if len(recipes) == 1:
            self._fit_member(
                recipe=next(iter(recipes.values())),
                x_context=x_context,
                y_context=y_context,
                generator=generator,
            )
            return

        weights = self._ensemble_weights(
            recipes=recipes,
            x_context=x_context,
            y_context=y_context,
            y=y.to_numpy(),
            folds=params["ensemble_folds"],
            iterations=params["ensemble_iterations"],
        )
        self._context = (x_context, y_context)
        self._members = [
            (index, recipe, weights[name])
            for index, (name, recipe) in enumerate(recipes.items())
            if weights[name] > 0
        ]

    def _generator(self, offset: int = 0) -> torch.Generator | None:
        seed = self.random_seed
        if not isinstance(seed, int):
            return None
        return torch.Generator(self._device).manual_seed(seed + offset)

    def _fit_member(
        self,
        *,
        recipe: sp.Recipe,
        x_context: torch.Tensor,
        y_context: torch.Tensor,
        generator: torch.Generator | None,
    ) -> None:
        with torch.amp.autocast(
            self._device.type,
            self.autocast_dtype,
            enabled=x_context.is_cuda,
        ):
            self.model.fit(
                x=x_context,
                y=y_context,
                recipe=recipe,
                num_estimators=None
                if self._expand_query
                else self._num_estimators,
                generator=generator,
            )

    def _ensemble_weights(
        self,
        *,
        recipes: dict[str, sp.Recipe],
        x_context: torch.Tensor,
        y_context: torch.Tensor,
        y: np.ndarray,
        folds: int,
        iterations: int,
    ) -> dict[str, float]:
        """Ensemble-selection weights from out-of-fold context predictions.

        Folds are stratified for classification; a fold whose training part
        lacks a class (single-row classes) is skipped.
        """
        splitter = (
            KFold(folds, shuffle=True, random_state=self.random_seed)
            if self.problem_type == REGRESSION
            else StratifiedKFold(
                folds, shuffle=True, random_state=self.random_seed
            )
        )
        num_classes = len(np.unique(y))
        truth: list[np.ndarray] = []
        oof: dict[str, list[np.ndarray]] = {name: [] for name in recipes}
        for rest, hold in splitter.split(np.zeros(len(y)), y):
            if (
                self.problem_type != REGRESSION
                and len(np.unique(y[rest])) < num_classes
            ):
                continue
            truth.append(y[hold])
            rest_dev = torch.as_tensor(rest, device=self._device)
            hold_dev = torch.as_tensor(hold, device=self._device)
            x_rest, y_rest = x_context[rest_dev], y_context[rest_dev]
            chunks = hold_dev.split(self._get_max_batch_size() or len(hold))
            for index, (name, recipe) in enumerate(recipes.items()):
                self._fit_member(
                    recipe=recipe,
                    x_context=x_rest,
                    y_context=y_rest,
                    generator=self._generator(index),
                )
                oof[name].append(
                    np.concatenate(
                        [self._predict_tensor(x_context[c]) for c in chunks]
                    )
                )
        if not truth:
            raise ValueError(
                "ensemble weights need a fold whose training part holds "
                "every class"
            )
        assert self.eval_metric is not None
        selection = EnsembleSelection(
            ensemble_size=iterations,
            problem_type=self.problem_type,
            metric=self.eval_metric,
        )
        selection.fit(
            predictions=[
                self._convert_proba_to_unified_form(np.concatenate(oof[name]))
                for name in recipes
            ],
            labels=np.concatenate(truth),
        )
        return dict(zip(recipes, selection.weights_.tolist()))

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
        if self._context is None:
            return self._convert_proba_to_unified_form(
                self._predict_tensor(x_query)
            )

        x_context, y_context = self._context
        total = None
        for index, recipe, weight in self._members:
            self._fit_member(
                recipe=recipe,
                x_context=x_context,
                y_context=y_context,
                generator=self._generator(index),
            )
            values = weight * self._predict_tensor(x_query)
            total = values if total is None else total + values
        assert total is not None
        return self._convert_proba_to_unified_form(total)

    def _predict_tensor(self, x_query: torch.Tensor) -> np.ndarray:
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
        return out.numerical[..., indices].float().cpu().numpy()

    def get_device(self) -> str:
        return str(next(self.model.parameters()).device)

    def _set_device(self, device: str) -> None:
        self.model.to(device)
        self._device = torch.device(device)
        if self._context is not None:
            x_context, y_context = self._context
            self._context = (x_context.to(device), y_context.to(device))

    def cleanup(self) -> None:
        self.model.clear()
        self._context = None
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

    def _set_default_params(self) -> None:
        super()._set_default_params()
        self._set_default_param_value("checkpoint", None)
        self._set_default_param_value("size", "large")
        self._set_default_param_value("numerical_missing", "nan")

    def _recipe(self, params: dict[str, Any]) -> sp.Recipe:
        return default_recipe(numerical_missing=params["numerical_missing"])

    def _recipes(self, params: dict[str, Any]) -> dict[str, sp.Recipe]:
        names = params["recipe_ensemble"]
        if not names:
            return super()._recipes(params)
        return {
            name: default_recipe(
                numerical_missing=params["numerical_missing"],
                **parse_recipe(name),
            )
            for name in names
        }

    def _create_model(
        self,
        task: Task,
        device: torch.device,
    ) -> sdm.models.KumoTabular:
        params = self._get_model_params()
        return sdm.models.KumoTabular(
            task=task,
            size=params["size"],
            device=device,
            checkpoint=params["checkpoint"],
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


def parse_recipe(name: str) -> dict[str, Any]:
    """default_recipe keyword arguments from a '+'-joined name.

    Tokens: a numeric transform (round_robin, identity, power, squash or
    quantile) and catshuffle<N> for ``shuffle_categories_max``.
    """
    kwargs: dict[str, Any] = {}
    for token in name.split("+"):
        if token in get_args(Normalize):
            kwargs["normalize"] = token
        elif token.startswith("catshuffle"):
            kwargs["shuffle_categories_max"] = int(
                token.removeprefix("catshuffle")
            )
        else:
            raise ValueError(f"unknown recipe token {token!r} in {name!r}")
    return kwargs


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
