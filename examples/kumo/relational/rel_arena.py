r"""Kumo model submission using RelArena's shared validation tuner.

Run one task from the repository root, after the setup in README.relarena.md::

    python -m examples.kumo.relational.rel_arena \
        --datasets rel-f1 --tasks driver-position --n-trials 2 \
        --search-space examples/kumo/relational/rel_arena_search_space.py \
        --output model-results.csv

Or call the same official runner from Python::

    from examples.kumo.relational.rel_arena import KumoModel
    from examples.kumo.relational.rel_arena_search_space import SEARCH_SPACE
    from relarena.runner import run_model_experiment

    result = run_model_experiment(
        KumoModel,
        "rel-f1",
        "driver-position",
        search_space=SEARCH_SPACE,
        n_trials=2,
        seed=0,
        cache_dir=".cache/relarena/rel-f1-driver-position",
    )
    print(result.tuned.test_score)

cache_dir is optional. Point it at precompute_text.py's output directory to
reuse frozen text embeddings; missing documents are encoded and cached. Omit
it to encode text on demand without a persistent cache. PCA remains fitted
separately on each training context in both cases.

rel_arena_search_space.py compares text OFF against context-fitted PCA32, with
all other parameters fixed across tasks. The system uses the same candidates.
Complete runtime has not been certified.
--search-space is optional; omit it to use the bundled policy. A custom Python
file must export SEARCH_SPACE and is executed, so use only trusted files.
RelArena scores every candidate on full validation, then refits and scores the
winner and default on TEST. The complete per-task budget includes preprocessing
and all trials/refits; a per-trial time limit is not a whole-task limit.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Literal, TypedDict, cast

import numpy as np
import pandas as pd
import torch
from examples.kumo.relational._relarena.lag import LAG_SPECS, RawEventLags
from examples.kumo.relational._relarena.text import ContextPCA, QwenDocuments
from examples.kumo.relational.rel_arena_search_space import (
    DEFAULT,
    SEARCH_SPACE,
    parse_search_space,
)
from relarena.model import RelArenaModel
from relarena.registry import register_model
from relbench.base import Database, EntityTask, Table, TaskType
from relbench.datasets import dataset_registry
from relbench.tasks import task_registry

import sdm
import sdm.processing as sp
from sdm.tensor import EnsembleTable

CONTEXT_ROWS = 10_000
ESTIMATORS = 8
QUERY_BATCH_SIZE = 1_000
RECENCY_POLICY = "seeded_cutoff_ties"
TEXT_TABLE_CHUNK_ROWS = 4_096


class _Config(TypedDict):
    num_neighbors: list[int]
    text_pca_components: int
    regression_transform: Literal["standard", "quantile"]
    cache_text: bool
    text_cache_max_bytes: int
    recent_context_pool: bool
    max_keys: int | None
    entity_timestamps: Literal["retain", "drop"]
    temporal_strategy: Literal["last", "uniform"]
    raw_event_lags: bool


class _SampleKwargs(TypedDict):
    task_link: dict[str, str]
    task_time_column: str
    num_neighbors: list[int]
    temporal_strategy: Literal["last", "uniform"]


class TargetQuantile(sp.Processor, sp.InvertibleMixin):
    """Use SDM's quantile mapping with V11 endpoint and median conventions."""

    handles_stypes = frozenset({sdm.Stype.numerical})
    requires_fit = True

    def __init__(self) -> None:
        super().__init__()
        self.mapping = sp.QuantileTransform(
            n_quantiles=1000,
            subsample=100_000,
            output_distribution="normal",
        )
        self.register_buffer("lower", torch.empty(0))
        self.register_buffer("upper", torch.empty(0))

    def _fit(
        self,
        table: sdm.TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        # Contexts have at most 10k rows, below the V11 100k subsample cap.
        self.mapping.fit(table, generator=generator)
        self.lower = table.numerical.amin(dim=-2, keepdim=True)
        self.upper = table.numerical.amax(dim=-2, keepdim=True)

    def _transform(self, table: sdm.TableTensor) -> sdm.TableTensor:
        output = self.mapping.transform(table)
        # V11 used Normal.icdf at the endpoints, not QuantileTransform's ndtri.
        normal = torch.distributions.Normal(0.0, 1.0)
        epsilon = table.numerical.new_tensor(
            1e-7 - torch.finfo(torch.float64).eps
        )
        numerical = torch.where(
            table.numerical <= self.lower,
            normal.icdf(epsilon),
            output.numerical,
        )
        numerical = torch.where(
            table.numerical >= self.upper,
            normal.icdf(1 - epsilon),
            numerical,
        )
        return output.replace_blocks(numerical=numerical)

    def _inverse_transform(self, table: sdm.TableTensor) -> sdm.TableTensor:
        if "q500" in table.columns[sdm.Stype.numerical]:
            table = table[["q500"]]
        return self.mapping.inverse_transform(table)


@contextmanager
def v11_precision() -> Iterator[None]:
    """Use V11 neural backend precision, restoring the caller's settings."""
    precision = torch.get_float32_matmul_precision()
    matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    cudnn_sdp = torch.backends.cuda.cudnn_sdp_enabled()
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_cudnn_sdp(False)
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(precision)
        torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
        torch.backends.cudnn.allow_tf32 = cudnn_tf32
        torch.backends.cuda.enable_cudnn_sdp(cudnn_sdp)


def context_members(frame: pd.DataFrame, *, seed: int) -> np.ndarray:
    """Draw eight independent contexts from all supplied training rows."""
    generator = torch.Generator().manual_seed(seed)
    return torch.stack(
        [
            torch.randperm(len(frame), generator=generator)[:CONTEXT_ROWS]
            for _ in range(ESTIMATORS)
        ]
    ).numpy()


def _recent_context_pool(
    frame: pd.DataFrame, time_col: str, *, seed: int, limit: int = 80_000
) -> pd.DataFrame:
    if len(frame) <= limit:
        return frame
    ordered = frame.sort_values(time_col, kind="stable")
    cutoff = ordered[time_col].iloc[-limit]
    newer = ordered[ordered[time_col] > cutoff]
    tied = ordered[ordered[time_col] == cutoff]
    generator = torch.Generator().manual_seed(seed)
    selected = torch.randperm(len(tied), generator=generator)[
        : limit - len(newer)
    ].numpy()
    return pd.concat([tied.iloc[selected], newer])


def table_stypes(table: Table, *, text: bool) -> dict[str, sdm.StypeLike]:
    """Infer the same supported columns for prediction and precomputation."""
    overrides = dict.fromkeys(table.fkey_col_to_pkey_table, "id")
    if table.pkey_col is not None:
        overrides[table.pkey_col] = "id"
    sample = table.df.head(10_000)
    dropped = [
        column
        for column in sample
        if column not in overrides
        and (
            column.startswith("Unnamed:")
            or sample[column].isna().all()
            or multicategorical(sample[column])
        )
    ]
    return sdm.infer_stypes(
        sample.drop(columns=dropped),
        overrides=overrides,
        text="infer" if text else "drop",
        unsupported="drop",
    )


def model_tables(db: Database, *, text: bool) -> dict[str, sdm.TableTensor]:
    """Infer supported columns, preserving relational keys and timestamps."""
    return {
        name: tensorize_table(
            frame=db.table_dict[name].df,
            stypes=table_stypes(db.table_dict[name], text=text),
        )
        for name in sorted(db.table_dict)
    }


def tensorize_table(
    frame: pd.DataFrame, stypes: dict[str, sdm.StypeLike]
) -> sdm.TableTensor:
    """Bound temporary string interleaving without changing inferred types."""
    if len(frame) <= TEXT_TABLE_CHUNK_ROWS or not any(
        sdm.Stype(stype) == sdm.Stype.text for stype in stypes.values()
    ):
        return sdm.TableTensor.from_pandas(df=frame, stypes=stypes)
    text_stypes = {
        name: stype
        for name, stype in stypes.items()
        if sdm.Stype(stype) == sdm.Stype.text
    }
    other_stypes = {
        name: stype
        for name, stype in stypes.items()
        if name not in text_stypes
    }
    # Keep whole-column key/null/category representations unchanged. Only
    # text needs chunking; row concatenation uses contiguous UTF8 buffers.
    text = torch.cat(
        [
            sdm.TableTensor.from_pandas(
                df=frame.iloc[start : start + TEXT_TABLE_CHUNK_ROWS],
                stypes=text_stypes,
            )
            for start in range(0, len(frame), TEXT_TABLE_CHUNK_ROWS)
        ],
        dim=0,
    )
    if not other_stypes:
        return cast(sdm.TableTensor, text)
    other = sdm.TableTensor.from_pandas(df=frame, stypes=other_stypes)
    return cast(sdm.TableTensor, torch.cat([other, text], dim=1))


def multicategorical(series: pd.Series) -> bool:
    """Match the V11 native-list and delimiter-list exclusion policy."""
    if any(
        isinstance(value, (list, tuple, set, np.ndarray))
        for value in series.dropna().iloc[:1000]
    ):
        return True
    values = series.iloc[:500].dropna()
    if (
        values.empty
        or not values.map(lambda value: isinstance(value, str)).all()
    ):
        return False
    counts: dict[str, int] = {}
    for character in "\n".join(values):
        if character in {";", ":", "|", "\t"}:
            counts[character] = counts.get(character, 0) + 1
    if not counts:
        return False
    delimiter = max(counts, key=counts.__getitem__)
    members = values.str.split(delimiter).explode().nunique()
    return values.nunique() > 1.5 * members and members <= 100


class KumoPredictor(RelArenaModel):
    """One configuration; RelArena owns selection and full split evaluation."""

    name = "sdm-kumo"
    supported_task_types = frozenset(
        {TaskType.BINARY_CLASSIFICATION, TaskType.REGRESSION}
    )

    def fit(
        self,
        task: EntityTask,
        db: Database,
        train_table: Table,
        val_table: Table | None,
        *,
        seed: int,
        time_limit: float | None = None,
    ) -> None:
        started = time.monotonic()
        self.deadline = None if time_limit is None else started + time_limit
        config = cast(_Config, {**DEFAULT, **self.config})
        unknown = self.config.keys() - DEFAULT.keys()
        if unknown:
            raise ValueError(
                f"Unsupported configuration keys: {sorted(unknown)}"
            )
        if config["entity_timestamps"] not in {"retain", "drop"}:
            raise ValueError("entity_timestamps must be retain or drop")
        if config["temporal_strategy"] not in {"last", "uniform"}:
            raise ValueError("temporal_strategy must be last or uniform")
        self.entity_timestamps = config["entity_timestamps"]
        self.device = torch.device("cuda")
        self.seed = seed
        self.binary = task.task_type == TaskType.BINARY_CLASSIFICATION
        # Only training rows provide labels, including on the outer refit.
        columns = [task.entity_col, task.time_col, task.target_col]
        frame = train_table.df[columns].copy().reset_index(drop=True)
        available_rows = len(frame)
        if config["recent_context_pool"]:
            frame = _recent_context_pool(frame, task.time_col, seed=seed)
        if self.binary:
            frame[task.target_col] = frame[task.target_col] == 1
        members = context_members(frame, seed=seed)
        self.context_metadata = {
            "requested_context_rows": CONTEXT_ROWS,
            "actual_context_rows_per_estimator": members.shape[1],
            "n_estimators": ESTIMATORS,
            "available_training_rows": available_rows,
            "eligible_context_pool_rows": len(frame),
            "recent_context_pool": config["recent_context_pool"],
            "recency_policy": RECENCY_POLICY,
            "max_keys": config["max_keys"],
            "entity_timestamps": config["entity_timestamps"],
            "temporal_strategy": config["temporal_strategy"],
        }
        selected = frame.iloc[members.flatten()].reset_index(drop=True)
        stypes = {
            task.entity_col: "id",
            task.time_col: "datetime",
            task.target_col: "categorical" if self.binary else "numerical",
        }
        self.raw_lags = None
        if config["raw_event_lags"]:
            # Registry metadata identifies formulas without loading any DB.
            for dataset_name, task_name in LAG_SPECS:
                if (
                    type(task) is task_registry[dataset_name][task_name][0]
                    and type(task.dataset) is dataset_registry[dataset_name][0]
                ):
                    self.raw_lags = RawEventLags(
                        db,
                        dataset=dataset_name,
                        task=task_name,
                        entity_col=task.entity_col,
                        time_col=task.time_col,
                    )
                    selected = pd.concat(
                        [selected, self.raw_lags.transform(selected)],
                        axis=1,
                    )
                    stypes.update(
                        dict.fromkeys(self.raw_lags.columns, "numerical")
                    )
                    break
        self.context_metadata["raw_event_lags"] = config["raw_event_lags"]
        self.context_metadata["raw_event_lag_columns"] = (
            len(self.raw_lags.columns) if self.raw_lags is not None else 0
        )
        context = sdm.TableTensor.from_pandas(
            df=selected,
            stypes=stypes,
        )
        context = context.unflatten(0, members.shape)
        components = config["text_pca_components"]
        # CPU data/sampler construction must remain outside inference_mode.
        tables = model_tables(db, text=components > 0)
        data = sdm.RelationalData(
            tables=tables,
            relationships=[
                {
                    "left_table": name,
                    "left_column": column,
                    "right_table": other,
                    "right_column": cast(str, db.table_dict[other].pkey_col),
                }
                for name in sorted(db.table_dict)
                for column, other in sorted(
                    db.table_dict[name].fkey_col_to_pkey_table.items()
                )
            ],
        )
        self.sampler = data.sampler(
            time_columns={
                name: table.time_col
                for name, table in sorted(db.table_dict.items())
                if table.time_col is not None
            }
        )
        self.sample_kwargs: _SampleKwargs = {
            "task_link": {
                "task_column": task.entity_col,
                "table": task.entity_table,
                "table_column": cast(
                    str, db.table_dict[task.entity_table].pkey_col
                ),
            },
            "task_time_column": task.time_col,
            "num_neighbors": config["num_neighbors"],
            "temporal_strategy": config["temporal_strategy"],
        }
        self.model = sdm.models.KumoRelational(
            task="classification" if self.binary else "regression",
            device=self.device,
            max_keys=config["max_keys"],
        )
        recipe = self.model.default_recipe()
        if config["regression_transform"] == "quantile":
            recipe.prepend_target(sp.StypeDispatch(numerical=TargetQuantile()))
        elif config["regression_transform"] != "standard":
            raise ValueError("Unknown regression transform")
        if components:
            cache_path = None
            if config["cache_text"] and self.cache.directory is not None:
                cache_path = self.cache.directory / "qwen-documents.sqlite"
            recipe.prepend_features(
                sp.StypeDispatch(
                    text=[
                        QwenDocuments(
                            self.device,
                            cache_path=cache_path,
                            max_vector_bytes=config["text_cache_max_bytes"],
                        ),
                        ContextPCA(components),
                    ]
                )
            )
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(seed)
            sampled = self.sampler(context, **self.sample_kwargs)
        sampled = sampled.to(self.device)
        self._check_time()
        with (
            v11_precision(),
            torch.amp.autocast("cuda", dtype=torch.float16),
        ):
            self.model.fit(
                x=sampled.task_table.drop_columns(task.target_col),
                y=sampled.task_table[task.target_col],
                related_tables=self._related_features(
                    sampled.related_tables, task.entity_table
                ),
                recipe=recipe,
                num_estimators=None,
                generator=torch.Generator(device=self.device).manual_seed(
                    seed
                ),
            )
        self._check_time()

    def _related_features(
        self, related: sdm.RelatedTables, entity_table: str
    ) -> sdm.RelatedTables:
        if self.entity_timestamps == "retain":
            return related
        # Only remove model features after sampling. The persistent sampler
        # retains every timestamp needed for temporal traversal.
        entity = related.tables[entity_table]
        if isinstance(entity, EnsembleTable):
            entity = entity.replace_groups(
                [group.drop_stypes(sdm.Stype.datetime) for group in entity]
            )
        else:
            entity = entity.drop_stypes(sdm.Stype.datetime)
        return related.replace_tables({**related.tables, entity_table: entity})

    def _check_time(self) -> None:
        if self.deadline is not None and time.monotonic() >= self.deadline:
            raise TimeoutError(
                "Trial budget exhausted; partial validation is not a result"
            )

    def predict(
        self, task: EntityTask, db: Database, table: Table
    ) -> np.ndarray:
        # Ignore query labels, even when the validation table carries them.
        query_frame = table.df[[task.entity_col, task.time_col]]
        stypes = {task.entity_col: "id", task.time_col: "datetime"}
        if self.raw_lags is not None:
            query_frame = pd.concat(
                [query_frame, self.raw_lags.transform(query_frame)], axis=1
            )
            stypes.update(dict.fromkeys(self.raw_lags.columns, "numerical"))
        query = sdm.TableTensor.from_pandas(
            df=query_frame,
            stypes=stypes,
        )
        predictions = []
        for index, batch in enumerate(query.split(QUERY_BATCH_SIZE)):
            self._check_time()
            with torch.random.fork_rng(devices=[]):
                torch.default_generator.manual_seed(self.seed + 3 + index)
                sampled = self.sampler(batch, **self.sample_kwargs)
            sampled = sampled.to(self.device)
            with (
                v11_precision(),
                torch.amp.autocast("cuda", dtype=torch.float16),
            ):
                output = self.model.predict(
                    x=EnsembleTable.from_table(
                        sampled.task_table, num_members=ESTIMATORS
                    ),
                    related_tables=sampled.related_tables.replace_tables(
                        {
                            name: EnsembleTable.from_table(
                                related, num_members=ESTIMATORS
                            )
                            for name, related in (
                                self._related_features(
                                    sampled.related_tables, task.entity_table
                                ).tables.items()
                            )
                        }
                    ),
                )
            column = "True" if self.binary else "q500"
            if (
                self.binary
                and column not in output.columns[sdm.Stype.numerical]
            ):
                prediction = output.numerical.new_zeros(len(batch))
            else:
                prediction = output[column].numerical.squeeze(-1)
            predictions.append(prediction.float().cpu().numpy())
        self._check_time()
        return np.concatenate(predictions).astype(np.float32, copy=False)


KumoModel = register_model(search_space=SEARCH_SPACE)(KumoPredictor)


if __name__ == "__main__":
    from relarena.cli import main

    search_space, arguments = parse_search_space(sys.argv[1:])
    register_model(search_space=search_space)(KumoModel)
    raise SystemExit(main(["--model", KumoModel.name, *arguments]))
