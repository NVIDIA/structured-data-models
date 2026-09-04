"""SDM tabular model adapter for TabArena and BeyondArena."""

import math
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, Self, cast

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
ModelFactory = Callable[
    [Task, torch.device, Path | None],
    sdm.models.ICLModel,
]


@dataclass(frozen=True)
class ModelConfig:
    name: str
    factory: ModelFactory
    num_estimators: int
    autocast_dtype: torch.dtype
    batch_size: int | None = None
    recipe_factory: Callable[[], sdm.Recipe] | None = None
    checkpoint_task: Task | None = None

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
    checkpoint_path: Path | None,
) -> sdm.models.TabICLv2:
    assert checkpoint_path is None
    return sdm.models.TabICLv2(task=task, device=device)


@lru_cache(maxsize=4)
def _create_kumo_tabular(
    task: Task,
    device: torch.device,
    checkpoint_path: Path | None,
) -> sdm.models.KumoTabular:
    return sdm.models.KumoTabular(
        task=task,
        device=device,
        checkpoint_path=checkpoint_path,
    )


@lru_cache(maxsize=1)
def _create_tabfm(
    task: Task,
    device: torch.device,
    checkpoint_path: Path | None,
) -> sdm.models.TabFM:
    assert checkpoint_path is None
    return sdm.models.TabFM(task=task, accept_license=True, device=device)


def _checkpoint_recipe() -> sdm.Recipe:
    return sp.Recipe(
        features=[
            sp.StypeDispatch(
                categorical=[
                    sp.AlignCategories(sort_by="value"),
                    sp.ToNumerical(missing_value=float("nan")),
                ],
            ),
            sp.StypeDispatch(
                numerical=[
                    sp.DropConstantColumns(),
                    sp.Standardize(epsilon=1e-6, ignore_nan=True),
                    sp.Clip(min_value=-100.0, max_value=100.0),
                    sp.ShuffleColumns(method="latin"),
                ],
            ),
        ],
        target=[
            sp.StypeDispatch(
                categorical=[
                    sp.AlignCategories(),
                    sp.ShuffleCategories(method="shift"),
                ],
                numerical=sp.Standardize(),
            ),
        ],
        output=[
            sp.ReduceEstimators(method="mean"),
            sp.TaskDispatch(
                classification=sp.Softmax(temperature=0.9),
            ),
        ],
    )


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
    "kumo-bmsg60zm": ModelConfig(
        name="KumoBmsg60zm",
        factory=_create_kumo_tabular,
        num_estimators=8,
        autocast_dtype=torch.bfloat16,
        batch_size=4096,
        recipe_factory=_checkpoint_recipe,
        checkpoint_task="regression",
    ),
    "kumo-3us8y132": ModelConfig(
        name="Kumo3us8y132",
        factory=_create_kumo_tabular,
        num_estimators=8,
        autocast_dtype=torch.bfloat16,
        batch_size=4096,
        checkpoint_task="regression",
    ),
    "kumo-2uirqewf": ModelConfig(
        name="Kumo2uirqewf",
        factory=_create_kumo_tabular,
        num_estimators=8,
        autocast_dtype=torch.bfloat16,
        batch_size=4096,
        recipe_factory=_checkpoint_recipe,
        checkpoint_task="regression",
    ),
    "tabfm": ModelConfig(
        name="TabFM",
        factory=_create_tabfm,
        num_estimators=32,
        autocast_dtype=torch.bfloat16,
    ),
}


def _column_limit(
    *,
    num_rows: int,
    max_columns: int | None,
    max_cells: int | None,
) -> int | None:
    if max_cells is None:
        return max_columns
    if max_cells < 1:
        raise ValueError("'max_cells' must be positive")

    cell_limit = max(1, max_cells // max(num_rows, 1))
    return cell_limit if max_columns is None else min(max_columns, cell_limit)


def _from_pandas(
    df: pd.DataFrame,
    *,
    stypes: dict[str, sdm.Stype | str],
    device: torch.device,
    replace_nonfinite: bool = False,
) -> sdm.TableTensor:
    table = sdm.TableTensor.from_pandas(
        df=df,
        stypes=stypes,
        device=device,
    )
    if replace_nonfinite and table.numerical.size(-1) > 0:
        numerical = table.numerical.masked_fill(
            ~table.numerical.isfinite(),
            float("nan"),
        )
        table = table.replace_blocks(numerical=numerical)
    return table


class SDMSystem(ExternalSystemModel):
    def __init__(
        self,
        *,
        model: str,
        checkpoint: str | None = None,
        checkpoint_path: str | None = None,
        max_context_size: int | None = None,
        max_columns: int | None = None,
        max_cells: int | None = None,
        batch_size: int | None = None,
        many_class: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._config = MODEL_CONFIGS[model]
        if checkpoint is not None and checkpoint_path is not None:
            raise ValueError("Only one checkpoint path may be provided")
        checkpoint_path = checkpoint_path or checkpoint
        self._checkpoint_path = (
            Path(checkpoint_path) if checkpoint_path is not None else None
        )
        self._max_context_size = max_context_size
        self._max_columns = max_columns
        self._max_cells = max_cells
        self._batch_size = batch_size
        self._many_class = many_class
        self._many_class_codebook: np.ndarray | None = None
        if max_cells is not None and max_cells < 1:
            raise ValueError("'max_cells' must be positive")
        if (
            self._checkpoint_path is not None
            and self._config.factory is not _create_kumo_tabular
        ):
            raise ValueError("A local checkpoint requires a KumoTabular model")
        if many_class and self._config.factory is not _create_kumo_tabular:
            raise ValueError("'many_class' requires a KumoTabular model")
        if (
            self._config.checkpoint_task is not None
            and self._checkpoint_path is None
        ):
            raise ValueError("This model requires a checkpoint path")

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
        if (
            self._config.checkpoint_task is not None
            and task != self._config.checkpoint_task
        ):
            raise ValueError(
                f"This model only supports {self._config.checkpoint_task}"
            )
        self.model = self._config.factory(
            task=task,
            device=self._device,
            checkpoint_path=self._checkpoint_path,
        )

        generator: torch.Generator | None = None
        if random_state is not None:
            generator = torch.Generator(self._device).manual_seed(random_state)

        inferred_stypes = sdm.infer_stypes(X)
        self.stypes = {
            name: str(stype) for name, stype in inferred_stypes.items()
        }
        x_context = _from_pandas(
            X,
            stypes=self.stypes,
            device=self._device,
            replace_nonfinite=self._checkpoint_path is not None,
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
        y_context = _from_pandas(
            y.rename(target_name).to_frame(),
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

        recipe = (
            self._config.recipe_factory()
            if self._config.recipe_factory is not None
            else None
        )
        if recipe is None and self._checkpoint_path is not None:
            recipe = _checkpoint_recipe()
        num_context_rows = min(len(X), max_context_size or len(X))
        max_columns = _column_limit(
            num_rows=num_context_rows,
            max_columns=self._max_columns,
            max_cells=self._max_cells,
        )
        if max_columns is not None:
            recipe = recipe or self.model.default_recipe()
            recipe.append_features(
                sp.SelectColumns(
                    max_columns,
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

    def _predict_batches(self, X: pd.DataFrame) -> sdm.TableTensor:
        x_query = _from_pandas(
            X,
            stypes=self.stypes,
            device=self._device,
            replace_nonfinite=self._checkpoint_path is not None,
        )
        if self.expand_query:
            num_estimators = self._config.num_estimators
            x_query = x_query.expand(num_estimators, *x_query.size())

        batch_size = (
            self._batch_size
            or self._config.batch_size
            or x_query.size(-2)
        )
        outs: list[sdm.TableTensor] = []
        for batch in x_query.split(batch_size, dim=-2):
            with torch.amp.autocast(
                self._device.type,
                self._config.autocast_dtype,
                enabled=batch.is_cuda,
            ):
                outs.append(self.model.predict(batch))
        if len(outs) == 1:
            return outs[0]
        tensors = cast(list[torch.Tensor], outs)
        return cast(sdm.TableTensor, torch.cat(tensors, dim=-2))

    def _predict(self, X: pd.DataFrame) -> pd.Series:
        out = self._predict_batches(X)
        values = out.numerical.float().mean(dim=-1).cpu().numpy()
        return pd.Series(values, index=X.index)

    def _predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        if self._many_class_codebook is not None:
            return self._predict_many_class_proba(X)

        probabilities = self._predict_batches(X).to_pandas()
        probabilities.index = X.index
        probabilities = probabilities.rename(
            columns=self._class_labels_by_key,
        )
        return probabilities.reindex(
            columns=tuple(self._class_labels_by_key.values()),
        )

    def _predict_many_class_proba(
        self,
        X: pd.DataFrame,
    ) -> pd.DataFrame:
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

                probabilities = self._predict_batches(X).to_pandas()
                symbols = np.asarray(probabilities.columns, dtype=np.int64)
                row_probabilities.append(
                    align_symbol_probabilities(
                        probabilities.to_numpy(),
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
