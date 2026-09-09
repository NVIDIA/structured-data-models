"""SDM tabular model adapter for TabArena and BeyondArena."""

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ClassVar, Literal, Self

import numpy as np
import pandas as pd
import torch
from autogluon.common.features.types import S_TEXT_EMBEDDING
from autogluon.tabular.models.abstract.abstract_torch_model import (
    AbstractTorchModel,
)

import sdm

Task = Literal["classification", "regression"]
ModelFactory = Callable[[Task, torch.device], sdm.models.ICLModel]


@dataclass(frozen=True)
class ModelConfig:
    name: str
    factory: ModelFactory
    num_estimators: int
    autocast_dtype: torch.dtype
    max_classes: int | None = None


def _create_tabiclv2(
    task: Task,
    device: torch.device,
) -> sdm.models.TabICLv2:
    return sdm.models.TabICLv2(task=task, device=device)


def _create_kumo_tabular(
    task: Task,
    device: torch.device,
) -> sdm.models.KumoTabular:
    return sdm.models.KumoTabular(task=task, device=device)


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
        max_classes=10,
    ),
    "tabfm": ModelConfig(
        name="TabFM",
        factory=_create_tabfm,
        num_estimators=32,
        autocast_dtype=torch.bfloat16,
        max_classes=10,
    ),
}


class SDMModel(AbstractTorchModel):
    """Adapt an SDM in-context model to the AutoGluon model interface."""

    ag_key = "SDM"
    ag_name = "SDM"
    seed_name = "random_state"
    _supported_problem_types: ClassVar[list[str]] = [
        "binary",
        "multiclass",
        "regression",
    ]
    _default_ag_args_ensemble_extra: ClassVar[dict[str, Any]] = {
        "fold_fitting_strategy": "sequential_local",
        "refit_folds": True,
    }
    default_num_gpus = 1
    default_resources_physical_cores_only = True
    minimum_num_gpus = 1
    gpu_strongly_recommended = True

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._config: ModelConfig | None = None
        self._device = torch.device("cpu")
        self._max_context_size: int | None = None
        self.stypes: dict[str, sdm.Stype] = {}
        self.text_embedding_columns: tuple[str, ...] = ()

    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        num_gpus: int = 0,
        **kwargs: Any,
    ) -> Self:
        params = dict(self._get_model_params())
        model_key = params.pop("model")
        self._max_context_size = params.pop("max_context_size", None)
        random_state_param = params.pop(self.seed_name, self.random_seed)
        random_state = (
            int(random_state_param) if random_state_param is not None else None
        )
        if params:
            names = ", ".join(sorted(params))
            raise ValueError(f"Unsupported SDM model parameters: {names}")

        self._config = MODEL_CONFIGS[model_key]
        self._device = torch.device(
            self._resolve_fit_device(num_gpus=num_gpus)
        )
        task: Task = (
            "regression"
            if self.problem_type == "regression"
            else "classification"
        )
        self.model = self._config.factory(task, self._device)

        generator: torch.Generator | None = None
        if random_state is not None:
            generator = torch.Generator(self._device).manual_seed(random_state)

        X = self.preprocess(X, y=y, is_train=True)
        if self._feature_metadata is not None:
            self.text_embedding_columns = tuple(
                self._feature_metadata.get_features(
                    required_special_types=[S_TEXT_EMBEDDING],
                )
            )
        self.stypes = sdm.infer_stypes(X)
        x_context = sdm.TableTensor.from_pandas(
            df=X,
            stypes=self.stypes,
            device=self._device,
        )

        target_name = str(y.name) if y.name is not None else "__target__"
        if self.problem_type == "regression":
            target_stype = "numerical"
        else:
            target_stype = "categorical"
            assert self.num_classes is not None
        y_context = sdm.TableTensor.from_pandas(
            df=y.rename(target_name).to_frame(),
            stypes={target_name: target_stype},
            device=self._device,
        )

        num_estimators = self._config.num_estimators
        max_context_size = self._max_context_size
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
        self.expand_query = num_estimators is None

        with torch.amp.autocast(
            self._device.type,
            self._config.autocast_dtype,
            enabled=x_context.is_cuda,
        ):
            self.model.fit(
                x=x_context,
                y=y_context,
                num_estimators=num_estimators,
                generator=generator,
            )

        return self

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
        if self.expand_query:
            assert self._config is not None
            num_estimators = self._config.num_estimators
            x_query = x_query.expand(num_estimators, *x_query.size())

        assert self._config is not None
        with torch.amp.autocast(
            self._device.type,
            self._config.autocast_dtype,
            enabled=x_query.is_cuda,
        ):
            out = self.model.predict(x_query)

        if self.problem_type == "regression":
            return (
                out.numerical.float()
                .mean(dim=-1)
                .cpu()
                .numpy()
                .astype(np.float32)
            )

        assert self.num_classes is not None
        probabilities = out.to_pandas().reindex(
            columns=[str(label) for label in range(self.num_classes)],
        )
        return self._convert_proba_to_unified_form(
            probabilities.to_numpy(dtype=np.float32),
        )

    def _set_default_params(self) -> None:
        defaults = {
            "model": "tabiclv2",
            "max_context_size": None,
        }
        for parameter, value in defaults.items():
            self._set_default_param_value(parameter, value)

    def get_device(self) -> str:
        if self.model is None:
            return self._device.type
        try:
            return next(self.model.parameters()).device.type
        except StopIteration:
            return self._device.type

    def _set_device(self, device: str) -> None:
        self._device = torch.device(device)
        if self.model is None:
            return
        self.model.to(self._device)
        # Module.to() does not move the SDM context cache.
        if self._device.type == "cpu" and self.model._cache is not None:
            self.model._cache = self.model._cache.cpu()

    def _more_tags(self) -> dict:
        return {"can_refit_full": True}
