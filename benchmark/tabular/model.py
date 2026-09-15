"""SDM model adapters for TabArena and BeyondArena."""

import abc
import logging
import math
from dataclasses import dataclass
from typing import Any, ClassVar, Literal, cast

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
from sdm.models.kumo.tabular.recipe import default_recipe, slot_recipe

Task = Literal["classification", "regression"]
logger = logging.getLogger(__name__)


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

    def _recipe(self, params: dict[str, Any]) -> sp.Recipe:
        return self.model.default_recipe()

    def _recipes(self, params: dict[str, Any]) -> dict[str, sp.Recipe]:
        """Candidate recipes; several turn on context-fitted slots."""
        return {"default": self._recipe(params)}

    def _set_default_params(self) -> None:
        self._set_default_param_value(
            "num_estimators",
            self.default_num_estimators,
        )
        self._set_default_param_value("max_context_size", None)
        self._set_default_param_value("max_columns", None)
        self._set_default_param_value("checkpoint", None)
        self._set_default_param_value("numerical_missing", "nan")
        self._set_default_param_value("recipe_ensemble", None)
        self._set_default_param_value("ensemble_folds", 3)
        self._set_default_param_value("ensemble_iterations", 20)

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

        recipes = self._recipes(params)
        for recipe in recipes.values():
            if params["max_columns"] is not None:
                for processor in recipe.features.modules():
                    if isinstance(processor, sp.SelectColumns):
                        processor.max_columns = params["max_columns"]

        self._x_context, self._y_context = x_context, y_context
        self._num_estimators_fit = num_estimators
        weights = dict.fromkeys(recipes, 1.0)
        if len(recipes) > 1 and not self._expand_query:
            weights = self._ensemble_weights(
                recipes=recipes,
                y=y.to_numpy(),
                folds=params["ensemble_folds"],
                iterations=params["ensemble_iterations"],
            )
        recipe = next(iter(recipes.values()))
        if len(recipes) > 1:
            recipe = self._slot_recipe(weights, params)
        self._fit_member(recipe, x_context, y_context)

    def _slot_recipe(
        self, weights: dict[str, float], params: dict[str, Any]
    ) -> sp.Recipe:
        raise NotImplementedError

    def _generator(self) -> torch.Generator | None:
        seed = self.random_seed
        if seed is None:
            return None
        return torch.Generator(self._device).manual_seed(seed)

    def _fit_member(
        self,
        recipe: sp.Recipe,
        x_context: sdm.TableTensor,
        y_context: sdm.TableTensor,
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
                num_estimators=self._num_estimators_fit,
                generator=self._generator(),
            )

    def _ensemble_weights(
        self,
        recipes: dict[str, sp.Recipe],
        y: np.ndarray,
        folds: int,
        iterations: int,
    ) -> dict[str, float]:
        """Caruana selection weights from k-fold out-of-fold context scores."""
        n = len(y)
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(0))
        parts = perm.chunk(folds)
        truth: list[np.ndarray] = []
        oof: dict[str, list[np.ndarray]] = {name: [] for name in recipes}
        for k in range(folds):
            hold = parts[k]
            rest = torch.cat([parts[j] for j in range(folds) if j != k])
            if len(np.unique(y[rest.numpy()])) < len(np.unique(y)):
                continue
            truth.append(y[hold.numpy()])
            hold_dev, rest_dev = hold.to(self._device), rest.to(self._device)
            for name, recipe in recipes.items():
                self._fit_member(
                    recipe,
                    self._x_context[rest_dev],
                    self._y_context[rest_dev],
                )
                oof[name].append(
                    self._predict_tensor(self._x_context[hold_dev])
                )
        y_true = np.concatenate(truth)
        preds = {name: np.concatenate(v) for name, v in oof.items()}

        def error(proba: np.ndarray) -> float:
            score = proba[:, 1] if proba.shape[1] == 2 else proba
            return float(self.eval_metric.error(y_true, score))

        chosen: list[str] = []
        current: np.ndarray | None = None
        for _ in range(iterations):
            best = min(
                recipes,
                key=lambda name: error(
                    preds[name]
                    if current is None
                    else (current * len(chosen) + preds[name])
                    / (len(chosen) + 1)
                ),
            )
            chosen.append(best)
            current = (
                preds[best]
                if current is None or len(chosen) == 1
                else (current * (len(chosen) - 1) + preds[best]) / len(chosen)
            )
        counts = {name: chosen.count(name) for name in recipes}
        return {name: counts[name] / iterations for name in recipes}

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
            x_query = cast(
                sdm.TableTensor,
                x_query.expand(self._num_estimators, *x_query.size()),
            )
        return self._finish_proba(self._predict_tensor(x_query))

    def _predict_tensor(self, x_query: sdm.TableTensor) -> np.ndarray:
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

    def _finish_proba(self, values: np.ndarray) -> np.ndarray:
        if self.problem_type == REGRESSION:
            return values
        return self._convert_proba_to_unified_form(values)

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

    def _create_model(
        self,
        task: Task,
        device: torch.device,
    ) -> sdm.models.KumoTabular:
        return sdm.models.KumoTabular(
            task=task,
            device=device,
            checkpoint=self._get_model_params()["checkpoint"],
        )

    def _recipe(self, params: dict[str, Any]) -> sp.Recipe:
        return default_recipe(numerical_missing=params["numerical_missing"])

    def _recipes(self, params: dict[str, Any]) -> dict[str, sp.Recipe]:
        names = params["recipe_ensemble"]
        if not names:
            return {"default": self._recipe(params)}
        return {name: default_recipe(**parse_recipe(name)) for name in names}

    def _slot_recipe(
        self, weights: dict[str, float], params: dict[str, Any]
    ) -> sp.Recipe:
        """One recipe whose estimator slots follow the candidate weights."""
        names = list(weights)
        raw = np.array([weights[n] for n in names]) * self._num_estimators
        counts = np.floor(raw).astype(int)
        for i in np.argsort(-(raw - counts))[
            : self._num_estimators - counts.sum()
        ]:
            counts[i] += 1
        slots = []
        for name, count in zip(names, counts):
            kwargs = parse_recipe(name)
            slots += [
                (
                    kwargs.get("numeric_transform", "round_robin"),
                    kwargs.get("shuffle_categories_max"),
                )
            ] * int(count)
        logger.log(20, "slots=%s", dict(zip(names, counts.tolist())))
        return slot_recipe(
            slots, numerical_missing=params["numerical_missing"]
        )


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
    """default_recipe kwargs from a '+'-joined name.

    First token is the NaN policy (nan, impute, mix); then identity, power
    or quantile, and catshuffle<N>.
    """
    tokens = name.split("+")
    kwargs: dict[str, Any] = {"numerical_missing": tokens[0]}
    for token in tokens[1:]:
        if token in ("identity", "power", "quantile"):
            kwargs["numeric_transform"] = token
        elif token.startswith("catshuffle"):
            kwargs["shuffle_categories_max"] = int(token[len("catshuffle") :])
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
