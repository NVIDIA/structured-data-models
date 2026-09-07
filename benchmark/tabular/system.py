"""SDM tabular model adapter for TabArena and BeyondArena."""

import math
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal, Self

import numpy as np
import pandas as pd
import torch
from autogluon.core.data import LabelCleaner
from tabarena.benchmark.exec_models.external import ExternalSystemModel

import sdm
import sdm.processing as sp
from benchmark.tabular._ecoc import (
    align_symbol_probabilities,
    decode_many_class_probabilities,
    many_class_codebook,
    many_class_estimator_count,
)

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
        num_estimators=32,
        autocast_dtype=torch.bfloat16,
    ),
}


class SDMSystem(ExternalSystemModel):
    def __init__(
        self,
        *,
        model: str,
        checkpoint: str | None = None,
        max_context_size: int | None = None,
        max_columns: int | None = None,
        batch_size: int | None = None,
        many_class: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._config = MODEL_CONFIGS[model]
        self._checkpoint = checkpoint
        self._max_context_size = max_context_size
        self._max_columns = max_columns
        self._batch_size = batch_size
        self._many_class = many_class
        self._many_class_codebook: np.ndarray | None = None
        if many_class and self._config.name != "KumoTabular":
            raise ValueError("'many_class' requires model 'kumo-tabular'")

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
        self._many_class_codebook = None
        self._device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        task: Task = (
            "regression" if problem_type == "regression" else "classification"
        )
        if self._checkpoint is None:
            self.model = self._config.factory(
                task=task,
                device=self._device,
            )
        else:
            if self._config.name != "KumoTabular":
                raise ValueError(
                    "A local checkpoint requires model 'kumo-tabular'"
                )
            self.model = sdm.models.KumoTabular.from_kumo_checkpoint(
                self._checkpoint,
                task=task,
                device=self._device,
            )

        generator: torch.Generator | None = None
        if random_state is not None:
            generator = torch.Generator(self._device).manual_seed(random_state)

        self.stypes = sdm.infer_stypes(X)
        x_context = sdm.TableTensor.from_pandas(
            df=X,
            stypes=self.stypes,
            device=self._device,
        )

        target_name = target_name or "__target__"
        if problem_type == "regression":
            target_stype = "numerical"
        else:
            target_stype = "categorical"
            cleaner = LabelCleaner.construct(problem_type=problem_type, y=y)
            self._class_labels_by_key = {
                str(label): label for label in cleaner.ordered_class_labels
            }
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

        recipe = None
        if self._max_columns is not None:
            recipe = self.model.default_recipe()
            recipe.append_features(
                sp.SelectColumns(
                    self._max_columns,
                    method="round_robin",
                )
            )

        if (
            self._many_class
            and task == "classification"
            and y_context.categorical.categories[0].numel() > 10
        ):
            if generator is None:
                generator = torch.Generator(self._device)
                generator.seed()
            n_classes = y_context.categorical.categories[0].numel()
            n_code_rows = many_class_estimator_count(n_classes, 10)
            seed = (
                random_state
                if random_state is not None
                else int(generator.initial_seed() % (2**32))
            )
            self._many_class_codebook = many_class_codebook(
                n_classes,
                10,
                n_code_rows,
                seed,
            )
            categories = y_context.categorical.categories[0].tolist()
            self._many_class_labels = tuple(
                self._class_labels_by_key[str(label)] for label in categories
            )
            self._many_class_x_context = x_context.cpu()
            self._many_class_y_context = y_context.cpu()
            self._many_class_num_estimators = num_estimators
            self._many_class_recipe = recipe
            self._many_class_generator_state = generator.get_state()
            self.model.clear()
            return self

        with torch.amp.autocast(
            self._device.type,
            self._config.autocast_dtype,
            enabled=x_context.is_cuda,
        ):
            self.model.fit(
                x=x_context,
                y=y_context,
                recipe=recipe,
                num_estimators=num_estimators,
                generator=generator,
            )

        return self

    def _predict(self, X: pd.DataFrame) -> pd.Series:
        x_query = sdm.TableTensor.from_pandas(
            df=X,
            stypes=self.stypes,
            device=self._device,
        )
        if self.expand_query:
            num_estimators = self._config.num_estimators
            x_query = x_query.expand(num_estimators, *x_query.size())

        outs = []
        for batch in x_query.split(self._batch_size or x_query.size(-2), -2):
            with torch.amp.autocast(
                self._device.type,
                self._config.autocast_dtype,
                enabled=batch.is_cuda,
            ):
                outs.append(self.model.predict(batch))
        out = torch.cat(outs, dim=-2) if len(outs) > 1 else outs[0]

        values = out.numerical.float().mean(dim=-1).cpu().numpy()
        return pd.Series(values, index=X.index)

    def _predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        if self._many_class_codebook is not None:
            return self._predict_many_class_proba(X)

        x_query = sdm.TableTensor.from_pandas(
            df=X,
            stypes=self.stypes,
            device=self._device,
        )
        if self.expand_query:
            num_estimators = self._config.num_estimators
            x_query = x_query.expand(num_estimators, *x_query.size())

        outs = []
        for batch in x_query.split(self._batch_size or x_query.size(-2), -2):
            with torch.amp.autocast(
                self._device.type,
                self._config.autocast_dtype,
                enabled=x_query.is_cuda,
            ):
                outs.append(self.model.predict(batch))
        out = torch.cat(outs, dim=-2) if len(outs) > 1 else outs[0]

        prob = out.to_pandas()
        prob.index = X.index
        prob = prob.rename(columns=self._class_labels_by_key)
        return prob.reindex(
            columns=tuple(self._class_labels_by_key.values()),
        )

    def _predict_many_class_proba(
        self,
        X: pd.DataFrame,
    ) -> pd.DataFrame:
        x_query = sdm.TableTensor.from_pandas(
            df=X,
            stypes=self.stypes,
            device=self._device,
        )
        if self.expand_query:
            x_query = x_query.expand(
                self._config.num_estimators,
                *x_query.size(),
            )

        row_probabilities = []
        try:
            for code_row in self._many_class_codebook:
                x_context = self._many_class_x_context.to(self._device)
                y_context = self._encode_many_class_target(code_row)
                generator = torch.Generator(self._device)
                generator.set_state(self._many_class_generator_state)

                with torch.amp.autocast(
                    self._device.type,
                    self._config.autocast_dtype,
                    enabled=x_context.is_cuda,
                ):
                    self.model.fit(
                        x=x_context,
                        y=y_context,
                        recipe=self._many_class_recipe,
                        num_estimators=self._many_class_num_estimators,
                        generator=generator,
                    )
                del x_context, y_context

                outs = []
                for batch in x_query.split(
                    self._batch_size or x_query.size(-2),
                    -2,
                ):
                    with torch.amp.autocast(
                        self._device.type,
                        self._config.autocast_dtype,
                        enabled=batch.is_cuda,
                    ):
                        outs.append(self.model.predict(batch))
                out = torch.cat(outs, dim=-2) if len(outs) > 1 else outs[0]
                prob = out.to_pandas()
                symbols = np.asarray(prob.columns, dtype=np.int64)
                row_probabilities.append(
                    align_symbol_probabilities(
                        prob.to_numpy(),
                        symbols,
                        alphabet_size=10,
                    )
                )
                self.model.clear()
        finally:
            self.model.clear()

        decoded = decode_many_class_probabilities(
            np.stack(row_probabilities),
            self._many_class_codebook,
        )
        return pd.DataFrame(
            decoded,
            index=X.index,
            columns=self._many_class_labels,
        ).reindex(columns=tuple(self._class_labels_by_key.values()))

    def _encode_many_class_target(
        self,
        code_row: np.ndarray,
    ) -> sdm.TableTensor:
        target = self._many_class_y_context.to(self._device)
        code = target.categorical.code
        if torch.any(code < 0):
            raise ValueError(
                "Many-class targets must not contain missing values"
            )
        mapping = torch.as_tensor(code_row, device=self._device)
        categorical = sdm.CategoricalTensor(
            code=mapping[code.to(torch.long)],
            categories=(torch.arange(10, device=self._device),),
        )
        return target.replace_blocks(categorical=categorical)
