# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SDM model adapters for TabArena and BeyondArena."""

import abc
import argparse
import copy
import math
from dataclasses import dataclass, replace
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
from benchmark.tabular.finetune import (
    FINETUNE_CONTEXT_FRAC,
    FINETUNE_EPOCHS,
    FINETUNE_ITERS_PER_EPOCH,
    FINETUNE_LR,
    FINETUNE_TRAIN_SIZE,
    FINETUNE_VAL_FRAC,
    full_finetune,
    kumo_small_binary_epochs,
)
from sdm.processing.execution import RecipeExecution

Task = Literal["classification", "regression"]
KumoTabularSize = Literal["small", "medium", "large"]

_FINETUNE_ARGS: dict[str, tuple[type, str]] = {
    "finetune_epochs": (int, "Fine-tuning epochs (only for '-ft' variants)."),
    "finetune_iters_per_epoch": (int, "Fine-tuning iterations per epoch."),
    "finetune_lr": (float, "Fine-tuning learning rate."),
    "finetune_train_size": (int, "Rows resampled per fine-tuning iteration."),
    "finetune_context_frac": (float, "Context fraction of each sample."),
    "finetune_val_frac": (float, "Held-out validation fraction of the pool."),
}


def add_finetune_args(parser: argparse.ArgumentParser) -> None:
    """Add the fine-tuning flags shared by TabArena/BeyondArena."""
    for name, (arg_type, help_text) in _FINETUNE_ARGS.items():
        parser.add_argument(f"--{name}", type=arg_type, help=help_text)


def finetune_config_overrides(args: argparse.Namespace) -> dict[str, Any]:
    """Explicitly-set fine-tuning flags, ready to merge into a model config."""
    return {
        name: value
        for name in _FINETUNE_ARGS
        if (value := getattr(args, name)) is not None
    }


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
    low_cardinality: ClassVar[Literal["off", "infer"]] = "off"

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
        self._set_default_param_value("kv_cache", False)
        self._set_default_param_value("estimator_batch_size", "auto")
        self._set_default_param_value("finetune", False)
        self._set_default_param_value("finetune_epochs", FINETUNE_EPOCHS)
        self._set_default_param_value(
            "finetune_iters_per_epoch", FINETUNE_ITERS_PER_EPOCH
        )
        self._set_default_param_value("finetune_lr", FINETUNE_LR)
        self._set_default_param_value(
            "finetune_train_size", FINETUNE_TRAIN_SIZE
        )
        self._set_default_param_value(
            "finetune_context_frac", FINETUNE_CONTEXT_FRAC
        )
        self._set_default_param_value("finetune_val_frac", FINETUNE_VAL_FRAC)

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
        self.stypes = sdm.infer_stypes(
            X,
            _low_cardinality=self.low_cardinality,
        )
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

        if params["finetune"]:
            finetune_epochs = kumo_small_binary_epochs(
                params["finetune_epochs"],
                is_kumo_small=self.ag_key == "SDM-KUMO-TABULAR-SMALL-FT",
                is_binary=self.problem_type == BINARY,
            )
            val_metric = full_finetune(
                self.model,
                x_context,
                y_context,
                task=task,
                max_epochs=finetune_epochs,
                iters_per_epoch=params["finetune_iters_per_epoch"],
                train_size=params["finetune_train_size"],
                context_frac=params["finetune_context_frac"],
                val_frac=params["finetune_val_frac"],
                lr=params["finetune_lr"],
                num_estimators=self._num_estimators,
                # Reuse max_context_size to cap fine-tuning's own validation
                # context the same way it caps the final fit()/predict() call.
                max_val_context_size=max_context_size,
                generator=generator,
            )
            print(
                f"[finetune] ag_key={self.ag_key} lr={params['finetune_lr']} "
                f"best_val_metric={val_metric}"
            )

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
        self._context_shape = x_context.shape[-2:]

        recipe = self.model.default_recipe()
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
                    estimator_batch_size=self._estimator_batch_size(x_context),
                    generator=generator,
                )
            return

        self._recipe_execution = RecipeExecution(recipe)
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
                outputs: list[sdm.TableTensor] = []
                size = self._estimator_batch_size(x_query)
                size = len(self._contexts) if size is None else size
                for start in range(0, len(self._contexts), size):
                    batch = [
                        context._replace(
                            x=cast(
                                sdm.TableTensor, context.x.to(self._device)
                            ),
                            y=cast(
                                sdm.TableTensor, context.y.to(self._device)
                            ),
                        )
                        for context in self._contexts[start : start + size]
                    ]
                    with torch.amp.autocast(
                        self._device.type,
                        self.autocast_dtype,
                        enabled=x_query.is_cuda,
                    ):
                        outputs += self.model._forward_members(
                            contexts=batch,
                            queries=queries[start : start + size],
                            estimator_batch_size=None,
                            generator=generator,
                        )

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

    def _estimator_batch_size(self, x: torch.Tensor) -> int | None:
        params = self._get_model_params()
        estimator_batch_size = params["estimator_batch_size"]
        if estimator_batch_size != "auto":
            return estimator_batch_size
        # Subsampled contexts can produce different cache shapes per estimator.
        if not x.is_cuda or self._expand_query:
            return 1

        num_rows, num_cols = self._context_shape
        if not params["kv_cache"]:
            # Uncached inference processes context and query rows together.
            num_rows += x.size(-2)
            if num_rows > 3_000 or num_rows * num_cols > 50_000:
                return 1
            return self._num_estimators

        num_rows = max(num_rows, x.size(-2))
        if num_rows > 2_000 or num_rows * num_cols >= 50_000:
            return 1
        return self._num_estimators

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


def _load_kumo_network(
    *,
    task: str,
    size: KumoTabularSize,
    device: torch.device,
) -> torch.nn.Module:
    return sdm.models.KumoTabular(
        task=task,
        size=size,
        device=device,
    ).models[task]


class SDMKumoTabularModel(SDMModel):
    size: ClassVar[KumoTabularSize]
    default_num_estimators = 16
    autocast_dtype = torch.float16
    # AutoGluon's feature generator hands binary columns over as integers.
    low_cardinality = "infer"
    # Bagged children are fit one at a time in this process, so they share the
    # pretrained network of their task through AutoGluon's registry.
    _default_ag_args_ensemble_extra: ClassVar[dict[str, Any]] = {
        "fold_fitting_strategy": "sequential_local",
    }
    shared_weights: ClassVar[SharedWeights] = SharedWeights(
        loader="benchmark.tabular.model:_load_kumo_network",
        key=("task", "size"),
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

    @classmethod
    def _create_model(
        cls,
        task: Task,
        device: torch.device,
    ) -> sdm.models.KumoTabular:
        model = sdm.models.KumoTabular(
            task=task,
            size=cls.size,
            pretrained=False,
            device="meta",
        )
        model.models[task] = _load_kumo_network(
            task=task,
            size=cls.size,
            device=device,
        )
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
                size=self.size,
                device=self._device,
            )


class SDMKumoTabularSmallModel(SDMKumoTabularModel):
    ag_key = "SDM-KUMO-TABULAR-SMALL"
    ag_name = "SDMKumoTabularSmall"
    size = "small"
    default_num_estimators = 8


class SDMKumoTabularMediumModel(SDMKumoTabularModel):
    ag_key = "SDM-KUMO-TABULAR-MEDIUM"
    ag_name = "SDMKumoTabularMedium"
    size = "medium"
    default_num_estimators = 8


class SDMKumoTabularLargeModel(SDMKumoTabularModel):
    ag_key = "SDM-KUMO-TABULAR-LARGE"
    ag_name = "SDMKumoTabularLarge"
    size = "large"


class _FinetunedMixin:
    """Mixin for a model's `-ft` variant: fine-tuning on by default."""

    def _set_default_params(self) -> None:
        super()._set_default_params()
        self.params["finetune"] = True


class SDMKumoTabularFinetunedModel(_FinetunedMixin, SDMKumoTabularLargeModel):
    ag_key = "SDM-KUMO-TABULAR-FT"
    ag_name = "SDMKumoTabularFT"
    # Fine-tuning mutates the network in place, so each fold needs its own
    # copy of the shared pretrained network rather than the same instance
    # every other bagged fold trains on top of.
    shared_weights: ClassVar[SharedWeights] = replace(
        SDMKumoTabularModel.shared_weights, copy_per_fit=True
    )


class SDMKumoTabularSmallFinetunedModel(
    _FinetunedMixin, SDMKumoTabularSmallModel
):
    ag_key = "SDM-KUMO-TABULAR-SMALL-FT"
    ag_name = "SDMKumoTabularSmallFT"
    shared_weights: ClassVar[SharedWeights] = replace(
        SDMKumoTabularModel.shared_weights, copy_per_fit=True
    )


class SDMTabICLv2FinetunedModel(_FinetunedMixin, SDMTabICLv2Model):
    ag_key = "SDM-TABICLV2-FT"
    ag_name = "SDMTabICLv2FT"


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


class SDMTabFMFinetunedModel(_FinetunedMixin, SDMTabFMModel):
    ag_key = "SDM-TABFM-FT"
    ag_name = "SDMTabFMFT"


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
    "kumo-tabular-small": ModelConfig(
        name="KumoTabular-Small",
        model_cls=SDMKumoTabularSmallModel,
    ),
    "kumo-tabular-medium": ModelConfig(
        name="KumoTabular-Medium",
        model_cls=SDMKumoTabularMediumModel,
    ),
    "kumo-tabular-large": ModelConfig(
        name="KumoTabular-Large",
        model_cls=SDMKumoTabularLargeModel,
    ),
    "kumo-tabular-ft": ModelConfig(
        name="KumoTabularFT",
        model_cls=SDMKumoTabularFinetunedModel,
    ),
    "tabiclv2-ft": ModelConfig(
        name="TabICLv2FT",
        model_cls=SDMTabICLv2FinetunedModel,
    ),
    "kumo-small": ModelConfig(
        name="KumoTabularSmall",
        model_cls=SDMKumoTabularSmallModel,
    ),
    "kumo-small-ft": ModelConfig(
        name="KumoTabularSmallFT",
        model_cls=SDMKumoTabularSmallFinetunedModel,
    ),
    "tabfm": ModelConfig(
        name="TabFM",
        model_cls=SDMTabFMModel,
    ),
    "tabfm-ft": ModelConfig(
        name="TabFMFT",
        model_cls=SDMTabFMFinetunedModel,
    ),
}
