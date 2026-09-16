import math
import sys
from functools import lru_cache

import numpy as np
import torch
from relarena.model import RelArenaModel
from relarena.registry import register_model
from relarena.search_space import SearchSpace
from relbench.base import Database, EntityTask, Table, TaskType

import sdm

KUMO_RELATIONAL_SPACE = SearchSpace(
    default_overrides={
        "context_size": 20_000,
        "num_neighbors": [8, 8],
        "num_estimators": 1,
        "lag_target": False,
        "ensemble_context": False,
    },
    fixed_grid=[
        {
            "context_size": context_size,
            "num_neighbors": num_neighbors,
            "num_estimators": num_estimators,
            "lag_target": lag_target,
            "ensemble_context": ensemble_context,
        }
        for context_size in [20_000]
        for num_neighbors in [
            [],
            [1, 1],
            [2, 2],
            [4, 4],
            [8, 8],
            [16, 16],
            [32, 32],
            # [64, 64],
            # [96, 96],
            # [128, 128],
        ]
        for num_estimators in [1, 8]
        for lag_target in [False]
        for ensemble_context in [True]
    ],
)


@lru_cache(maxsize=1)
def get_sampler(
    db: Database,
    task: EntityTask,
    train_table: Table,
    lag_targets: bool,
) -> sdm.relational.RelationalSampler:
    print("GET SAMPLER")

    tables = {
        name: sdm.TableTensor.from_pandas(
            df=table.df,
            stypes=sdm.infer_stypes(
                table.df.head(10_000),
                overrides={
                    table.pkey_col: "id",
                    **dict.fromkeys(table.fkey_col_to_pkey_table, "id"),
                },
                text="drop",
                unsupported="drop",
            ),
        )
        for name, table in db.table_dict.items()
    }
    if lag_targets:
        history = train_table.df.copy()
        history[task.time_col] -= task.timedelta
        tables["history"] = sdm.TableTensor.from_pandas(
            history,
            stypes={
                task.entity_col: "id",
                task.time_col: "datetime",
                task.target_col: "numerical"
                if task.task_type == TaskType.REGRESSION
                else "categorical",
            },
        )
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
    if lag_targets:
        relationships.append(
            {
                "left_table": "history",
                "left_column": task.entity_col,
                "right_table": task.entity_table,
                "right_column": db.table_dict[task.entity_table].pkey_col,
            }
        )

    return sdm.RelationalData(tables, relationships).sampler(
        time_columns={
            name: table.time_col
            for name, table in db.table_dict.items()
            if table.time_col is not None
        }
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

        self.expand_query = False
        context_size = self.config["context_size"]
        num_estimators = self.config["num_estimators"]
        generator = torch.Generator().manual_seed(seed)
        if self.config["ensemble_context"] and len(context) > context_size:
            repeats = math.ceil(context_size * num_estimators / len(context))
            perm = torch.cat(
                [
                    torch.randperm(len(context), generator=generator)
                    for _ in range(repeats)
                ]
            )
            context = context[perm[: context_size * num_estimators]]
            if num_estimators > 1:
                print("EXPAND CONTEXT/QUERY")
                context = context.unflatten(0, (num_estimators, context_size))
                num_estimators = None
                self.expand_query = True
        else:
            perm = torch.randperm(len(context), generator=generator)
            context = context[perm[:context_size]]

        self.sampler = get_sampler(
            db=db,
            task=task,
            train_table=train_table,
            lag_targets=self.config["lag_target"],
        )
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
                num_estimators=num_estimators,
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
        if self.expand_query:
            query = query.expand(self.config["num_estimators"], *query.size())

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
            out = out["1"].numerical.squeeze(-1)
            # out = out["True"].numerical.squeeze(-1)

        return out.cpu().numpy()


if __name__ == "__main__":
    from relarena.cli import main

    main(["--model", KumoRelationalModel.name, *sys.argv[1:]])
