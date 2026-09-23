# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SDM model adapters for TabArena and BeyondArena."""

import abc
import argparse
import math
from dataclasses import dataclass
from typing import Any, ClassVar, Literal

import numpy as np
import pandas as pd
import torch
from autogluon.core.constants import BINARY, MULTICLASS, REGRESSION
from autogluon.tabular.models.abstract.abstract_torch_model import (
    AbstractTorchModel,
)

import sdm
import sdm.processing as sp
from benchmark.tabular.finetune import full_finetune
from benchmark.tabular.kumo import load_kumo_tabular

Task = Literal["classification", "regression"]

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
        self._set_default_param_value("finetune", False)
        self._set_default_param_value("finetune_epochs", 75)
        self._set_default_param_value("finetune_iters_per_epoch", 10)
        self._set_default_param_value("finetune_lr", 1e-6)
        self._set_default_param_value("finetune_train_size", 10_000)
        self._set_default_param_value("finetune_context_frac", 0.8)
        self._set_default_param_value("finetune_val_frac", 0.2)

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

        if params["finetune"]:
            finetune_epochs = params["finetune_epochs"]
            if (
                self.ag_key == "SDM-KUMO-TABULAR-SMALL-FT"
                and self.problem_type == BINARY
            ):
                # Empirically found to need fewer epochs than the shared
                # default to avoid overfitting on binary classification.
                finetune_epochs = 50
            full_finetune(
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
        return load_kumo_tabular(task=task, size="large", device=device)


class SDMTabICLv2FinetunedModel(SDMTabICLv2Model):
    ag_key = "SDM-TABICLV2-FT"
    ag_name = "SDMTabICLv2FT"

    def _set_default_params(self) -> None:
        super()._set_default_params()
        self.params["finetune"] = True


class SDMKumoTabularSmallModel(SDMKumoTabularModel):
    ag_key = "SDM-KUMO-TABULAR-SMALL"
    ag_name = "SDMKumoTabularSmall"

    @staticmethod
    def _create_model(
        task: Task,
        device: torch.device,
    ) -> sdm.models.KumoTabular:
        return load_kumo_tabular(task=task, size="small", device=device)


class SDMKumoTabularSmallFinetunedModel(SDMKumoTabularSmallModel):
    ag_key = "SDM-KUMO-TABULAR-SMALL-FT"
    ag_name = "SDMKumoTabularSmallFT"

    def _set_default_params(self) -> None:
        super()._set_default_params()
        self.params["finetune"] = True


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
}
