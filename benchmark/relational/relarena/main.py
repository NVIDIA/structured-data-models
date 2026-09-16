r"""Kumo model submission using RelArena's shared validation tuner.

Run one task from the repository root, after the setup in README.relarena.md::

    python -m examples.kumo.relational.rel_arena_model \
        --datasets rel-f1 --tasks driver-position --n-trials 2 \
        --search-space examples/kumo/relational/rel_arena_search_space.py \
        --output model-results.csv

Or call the same official runner from Python::

    from examples.kumo.relational.rel_arena_model import KumoModel
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

Set cache_dir to an empty directory to cache newly computed Qwen embeddings
across tuning, refitting, and test prediction. No precomputation is required.
Omit it to disable persistent caching. PCA remains fitted separately on each
training context. Alternatively, point cache_dir at precompute_text.py's output
directory to reuse existing embeddings; missing documents are encoded and
cached.

rel_arena_search_space.py compares text OFF against context-fitted PCA32, with
all other parameters fixed across tasks. The system uses the same candidates.
Complete runtime has not been certified.
--search-space is optional; omit it to use the bundled policy. A custom Python
file must export SEARCH_SPACE and is executed, so use only trusted files.
RelArena scores every candidate on full validation, then refits and scores the
winner and default on TEST. The complete per-task budget includes preprocessing
and all trials/refits; a per-trial time limit is not a whole-task limit.
"""

import sys

import numpy as np
import torch

# from examples.kumo.relational._relarena.lag import LAG_SPECS, RawEventLags
# from examples.kumo.relational._relarena.text import ContextPCA, QwenDocuments
# from examples.kumo.relational.rel_arena_search_space import (
#     DEFAULT,
#     SEARCH_SPACE,
#     parse_search_space,
# )
from relarena.model import RelArenaModel
from relarena.registry import register_model
from relarena.search_space import SearchSpace
from relbench.base import Database, EntityTask, Table, TaskType

# from relbench.base import Database, EntityTask, Table, TaskType
# from relbench.datasets import dataset_registry
# from relbench.tasks import task_registry
import sdm

KUMO_RELATIONAL_SPACE = SearchSpace(
    default_overrides={
        "context_size": 20_000,
        "num_neighbors": [8, 8],
        "num_estimators": 1,
    },
    fixed_grid=[
        {
            "context_size": context_size,
            "num_neighbors": num_neighbors,
            "num_estimators": num_estimators,
            "lag_target": lag_target,
        }
        for context_size in [20_000]
        for num_neighbors in [
            # [],
            # [1, 1],
            # [2, 2],
            # [4, 4],
            # [8, 8],
            # [16, 16],
            # [32, 32],
            # [64, 64],
            # [96, 96],
            [128, 128],
        ]
        for num_estimators in [1, 8]
        for lag_target in [False]
    ],
)


@register_model(search_space=KUMO_RELATIONAL_SPACE)
class KumoRelationalModel(RelArenaModel):
    name = "kumo-relational"

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

        history = train_table.df.copy()
        history[task.time_col] -= task.timedelta

        for name, table in db.table_dict.items():
            columns = list(table.df.columns)
            bla = [
                column for column in columns if column.startswith("Unnamed")
            ]
            if len(bla) > 0:
                print(name, bla)

        stypes = {
            name: sdm.infer_stypes(
                table.df.head(10_000),
                overrides={
                    table.pkey_col: "id",
                    **dict.fromkeys(table.fkey_col_to_pkey_table, "id"),
                },
                text="drop",
                unsupported="drop",
            )
            for name, table in db.table_dict.items()
        }
        tables = {
            name: sdm.TableTensor.from_pandas(table.df, stypes[name])
            for name, table in db.table_dict.items()
        }
        if self.config["lag_target"]:
            tables["history"] = sdm.TableTensor.from_pandas(history, stypes={
                task.entity_col: "id",
                task.time_col: "datetime",
                task.target_col: "numerical"
                if task.task_type == TaskType.REGRESSION
                else "categorical",
            })
        relationships = [
            {
                "left_table": name,
                "left_column": column,
                "right_table": other,
                "right_column": db.table_dict[other].pkey_col,
            }
            for name, table in db.table_dict.items()
            for column, other in table.fkey_col_to_pkey_table.items()
        ]
        if self.config["lag_target"]:
            relationships.append({
                "left_table": "history",
                "left_column": task.entity_col,
                "right_table": task.entity_table,
                "right_column": db.table_dict[task.entity_table].pkey_col,
            })

        self.sampler = sdm.RelationalData(tables, relationships).sampler(
            time_columns={
                name: table.time_col
                for name, table in db.table_dict.items()
                if table.time_col is not None
            }
        )

        context = sdm.TableTensor.from_pandas(
            df=train_table.df,
            stypes={
                task.entity_col: "id",
                task.time_col: "datetime",
                task.target_col: "numerical"
                if task.task_type == TaskType.REGRESSION
                else "categorical",
            },
        )
        print('context', len(context))

        generator = torch.Generator().manual_seed(seed)
        perm = torch.randperm(len(context), generator=generator)
        context = context[perm[: self.config["context_size"]]]

        context, related_tables = self.sampler(
            context,
            task_link={
                "task_column": task.entity_col,
                "table": task.entity_table,
                "table_column": db.table_dict[task.entity_table].pkey_col,
            },
            num_neighbors=self.config["num_neighbors"],
            task_time_column=task.time_col,
        ).cuda()

        self.model = sdm.models.KumoRelational(
            task="regression"
            if task.task_type == TaskType.REGRESSION
            else "classification",
            device="cuda",
        )
        generator = torch.Generator(device="cuda").manual_seed(seed)
        with torch.amp.autocast("cuda", torch.float16, enabled=True):
            self.model.fit(
                x=context.drop_columns(task.target_col),
                y=context[task.target_col],
                related_tables=related_tables,
                num_estimators=self.config["num_estimators"],
                generator=generator,
            )

    def predict(
        self,
        task: EntityTask,
        db: Database,
        table: Table,
    ) -> np.ndarray:

        query = sdm.TableTensor.from_pandas(
            df=table.df,
            stypes={
                task.entity_col: "id",
                task.time_col: "datetime",
            },
        )
        print('query', len(query))


        outs = []
        for batch in query.split(10_000, dim=-2):
            batch, related_tables = self.sampler(
                batch,
                task_link={
                    "task_column": task.entity_col,
                    "table": task.entity_table,
                    "table_column": db.table_dict[task.entity_table].pkey_col,
                },
                num_neighbors=self.config["num_neighbors"],
                task_time_column=task.time_col,
            ).cuda()

            with torch.amp.autocast("cuda", torch.float16, enabled=True):
                outs.append(self.model.predict(batch, related_tables))
        out = torch.cat(outs, dim=-2)

        if task.task_type == TaskType.BINARY_CLASSIFICATION:
            # out = out["1"].numerical.squeeze(-1)
            out = out["True"].numerical.squeeze(-1)

        return out.cpu().numpy()


if __name__ == "__main__":
    from relarena.cli import main

    main(["--model", KumoRelationalModel.name, *sys.argv[1:]])
