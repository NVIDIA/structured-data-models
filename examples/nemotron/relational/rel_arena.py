"""Benchmark NemotronRelational with RelArena.

Without arguments, this runs every task in the seven RelBench v1 datasets
covered by RelArena. Pass ``--dataset`` to run one dataset or both
``--dataset`` and ``--task`` to run one task.

Examples:
    python rel_arena.py
    python rel_arena.py --dataset rel-f1 --task driver-dnf
    python rel_arena.py --dataset rel-f1 --num_neighbors 32
    python rel_arena.py --dataset rel-f1 --num_neighbors 16 16

Each ``--num_neighbors`` value configures one hop: ``32`` is one hop,
``16 16`` is two hops, and ``16 16 8`` is three hops.
"""

import argparse
from typing import Any, cast

import numpy as np
import relarena
import torch
import tqdm
from relarena.search_space import SearchSpace
from relbench.base import Database, EntityTask, Table, TaskType

import sdm

parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter,
)
parser.add_argument("--dataset", choices=relarena.RELBENCH_V1_DATASETS)
parser.add_argument("--task")
parser.add_argument("--context_size", type=int, default=10_000)
parser.add_argument("--batch_size", type=int, default=1000)
parser.add_argument("--num_neighbors", type=int, nargs="+", default=[16, 16])
parser.add_argument("--num_estimators", type=int, default=1)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument(
    "--no_test",
    action="store_true",
    help="only fit on train and evaluate on validation",
)
args = parser.parse_args()
if args.task and not args.dataset:
    parser.error("'--task' requires '--dataset'")


class NemotronRelationalModel(relarena.RelArenaModel):
    """Adapt NemotronRelational to RelArena's model interface."""

    name = "nemotron-relational"

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
        """Fit on the labels and censored database supplied by RelArena."""
        torch.manual_seed(seed)
        self._device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._binary = task.task_type == TaskType.BINARY_CLASSIFICATION

        tables = {}
        for name, table in db.table_dict.items():
            overrides = dict.fromkeys(table.fkey_col_to_pkey_table, "id")
            if table.pkey_col is not None:
                overrides[table.pkey_col] = "id"
            tables[name] = sdm.TableTensor.from_pandas(
                df=table.df,
                stypes=sdm.infer_stypes(
                    table.df.head(10_000),
                    overrides=overrides,
                    text="drop",
                    unsupported="drop",
                ),
            )

        data = sdm.RelationalData(
            tables=tables,
            relationships=[
                {
                    "left_table": left_table,
                    "left_column": left_column,
                    "right_table": right_table,
                    "right_column": cast(
                        str, db.table_dict[right_table].pkey_col
                    ),
                }
                for left_table, table in db.table_dict.items()
                for left_column, right_table in (
                    table.fkey_col_to_pkey_table.items()
                )
            ],
        )
        time_columns = {
            name: table.time_col
            for name, table in db.table_dict.items()
            if table.time_col is not None
        }
        self._sampler = data.sampler(time_columns)
        self._sample_kwargs: dict[str, Any] = {
            "task_link": {
                "task_column": task.entity_col,
                "table": task.entity_table,
                "table_column": cast(
                    str, db.table_dict[task.entity_table].pkey_col
                ),
            },
            "num_neighbors": self.config["num_neighbors"],
            "task_time_column": task.time_col,
        }

        context_df = train_table.df.copy()
        if self._binary:
            labels = sorted(context_df[task.target_col].dropna().unique())
            context_df[task.target_col] = (
                context_df[task.target_col] == labels[-1]
            )
        context = sdm.TableTensor.from_pandas(
            df=context_df,
            stypes={
                task.entity_col: "id",
                task.time_col: "datetime",
                task.target_col: (
                    "categorical" if self._binary else "numerical"
                ),
            },
        )
        context = context[
            torch.randperm(len(context))[: self.config["context_size"]]
        ]

        self._model = sdm.models.NemotronRelational(device=self._device)
        context, related_tables = self._sampler(
            context,
            **self._sample_kwargs,
        ).to(self._device)
        with torch.amp.autocast(
            self._device.type,
            torch.bfloat16,
            enabled=self._device.type == "cuda",
        ):
            self._model.fit(
                x=context.drop_columns(task.target_col),
                y=context[task.target_col],
                related_tables=related_tables,
                num_estimators=self.config["num_estimators"],
            )

    def predict(
        self,
        task: EntityTask,
        db: Database,
        table: Table,
    ) -> np.ndarray:
        """Predict in batches and return RelArena's one-dimensional output."""
        query = sdm.TableTensor.from_pandas(
            df=table.df[[task.entity_col, task.time_col]],
            stypes={
                task.entity_col: "id",
                task.time_col: "datetime",
            },
        )
        predictions = []
        for batch in tqdm.tqdm(query.split(self.config["batch_size"])):
            with torch.amp.autocast(
                self._device.type,
                torch.bfloat16,
                enabled=self._device.type == "cuda",
            ):
                out = self._model.predict(
                    *self._sampler(
                        batch,
                        **self._sample_kwargs,
                    ).to(self._device)
                )
            pred = (
                out["True"].numerical.squeeze(-1)
                if self._binary
                else out["q500"].numerical.squeeze(-1)
            )
            predictions.append(pred.float().cpu().numpy())
        return np.concatenate(predictions)


config = {
    "context_size": args.context_size,
    "batch_size": args.batch_size,
    "num_neighbors": args.num_neighbors,
    "num_estimators": args.num_estimators,
}
search_space = SearchSpace(default_overrides=config)
datasets = (
    [args.dataset] if args.dataset else list(relarena.RELBENCH_V1_DATASETS)
)
specs = relarena.list_entity_tasks(datasets)
if args.task:
    specs = [spec for spec in specs if spec.task == args.task]

for spec in specs:
    summary = relarena.run_experiment(
        NemotronRelationalModel,
        spec.dataset,
        spec.task,
        search_space=search_space,
        seed=args.seed,
        n_trials=0,
        cache_predictions=False,
        evaluate_test=not args.no_test,
    )
    trial = cast(relarena.TrialResult, summary.tuned)
    if not args.no_test and trial.test_score is None:
        raise RuntimeError(f"{spec.dataset}/{spec.task}: test refit failed")
    test_score = (
        f"{trial.test_score:.4f}"
        if trial.test_score is not None
        else "not evaluated"
    )
    print(
        f"{spec.dataset}/{spec.task} {summary.metric_name}: "
        f"val={trial.val_score:.4f}, test={test_score}"
    )
