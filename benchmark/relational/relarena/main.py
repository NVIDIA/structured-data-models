import argparse
import math
import sys
import tqdm
from functools import lru_cache
from itertools import product

import numpy as np
import pandas as pd
import torch
from relarena.model import RelArenaModel
from relarena.registry import register_model
from relarena.search_space import SearchSpace, TaskStats
from relbench.base import Database, EntityTask, Table, TaskType

import sdm

DEFAULT_CONFIG = {}  # TODO
NUM_NEIGHBORS: list[int] | None = None
NUM_LAGS: int | None = None


def add_lag_target_features(  # TODO Make more efficient.
    df: pd.DataFrame,
    history: pd.DataFrame,
    task: EntityTask,
    *,
    num_lags: int,
) -> pd.DataFrame:
    if num_lags == 0:
        return df

    entity_col = task.entity_col
    time_col = task.time_col
    target_col = task.target_col
    history_time_col = "__history_time__"
    lookup_time_col = "__lookup_time__"
    row_col = "__row__"

    out = df.copy()
    right = history[[entity_col, time_col, target_col]].rename(
        columns={time_col: history_time_col}
    )
    right = right.sort_values([history_time_col, entity_col])

    left = pd.DataFrame(
        {
            entity_col: out[entity_col].to_numpy(),
            lookup_time_col: out[time_col].to_numpy(),
            row_col: np.arange(len(out)),
        }
    )

    for lag in range(1, num_lags + 1):
        lag_col = f"{target_col}_lag_{lag}"
        if task.task_type == TaskType.REGRESSION:
            out[lag_col] = np.nan
        else:
            out[lag_col] = pd.Series(pd.NA, index=out.index, dtype="object")

        left = left.sort_values([lookup_time_col, entity_col])
        merged = pd.merge_asof(
            left,
            right,
            by=entity_col,
            left_on=lookup_time_col,
            right_on=history_time_col,
            direction="backward",
            allow_exact_matches=False,
        )

        mask = merged[history_time_col].notna()
        if mask.any():
            rows = merged.loc[mask, row_col].to_numpy()
            out.loc[out.index[rows], lag_col] = merged.loc[
                mask, target_col
            ].to_numpy()

        left = merged.loc[mask, [entity_col, row_col, history_time_col]]
        left = left.rename(columns={history_time_col: lookup_time_col})

    return out


def search_space(stats: TaskStats) -> SearchSpace:
    if stats.num_train_nodes < 2_000:
        # print("Low Data Regime", stats.num_train_nodes)
        # Prevent overfitting in low-data regimes:
        num_neighbors = [[], [1, 1], [8, 8]]
        num_estimators = [1]
    else:
        # print("Large Data Regime", stats.num_train_nodes)
        num_neighbors = [[], [1, 1], [8, 8], [16, 16], [32, 32], [64, 64], [96, 96], [128, 128]]
        # num_neighbors = [[8, 8], [16, 16], [32, 32], [64, 64]]
        # num_neighbors = [[4, 4], [8, 8], [16, 16], [32, 32], [4], [8]u
        # num_neighbors = [[32, 32]]
        # num_neighbors = [[32, 32], [48, 48], [64, 64]]
        # num_neighbors = [[128, 128],
        #                   [128]]
        # num_neighbors = [[32, 32]]
        num_estimators = [8]

    num_estimators = [8]
    context_size = [20_00]

    if NUM_LAGS is not None:
        num_lags = [NUM_LAGS]
    else:
        num_lags = [0, 10, 20]

    if NUM_NEIGHBORS is not None:
        num_neighbors = [NUM_NEIGHBORS]
    else:
        num_neighbors = [[32, 32]]

    if context_size[0] < stats.num_train_nodes:
        ensemble_context = [True]
    else:
        ensemble_context = [False]

    keys = (
        "context_size",
        "num_neighbors",
        "num_estimators",
        "num_lags",
        "ensemble_context",
    )

    fixed_grid = [
        dict(zip(keys, values, strict=True))
        for values in product(
            context_size,
            num_neighbors,
            num_estimators,
            num_lags,
            ensemble_context,
        )
    ]

    return SearchSpace(
        default_overrides={},
        fixed_grid=fixed_grid,
    )


@lru_cache(maxsize=1)
def get_sampler(db: Database) -> sdm.relational.RelationalSampler:
    tables = {}
    for name, table in db.table_dict.items():
        stypes=sdm.infer_stypes(
            table.df.head(10_000),
            overrides={
                table.pkey_col: "id",
                **dict.fromkeys(table.fkey_col_to_pkey_table, "id"),
            },
            text="drop",
            unsupported="drop",
        )
        # print(name, "========================")
        # for col, stype in stypes.items():
        #     print(col, stype)
        # if name == 'races':
        #     del stypes['name']
        if name == 'drivers':
            # del stypes["code"]
            del stypes["forename"]
            del stypes["surname"]
        # if name == "customer":
        #     del stypes["customer_name"]
        # print(name)
        # for col, stype in stypes.items():
        #     print(col, stype)
        tables[name] = sdm.TableTensor.from_pandas(table.df, stypes)

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

    return sdm.RelationalData(tables, relationships).sampler(
        time_columns={
            name: table.time_col
            for name, table in db.table_dict.items()
            if table.time_col is not None
        }
    )


@register_model(search_space=search_space)
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

        # torch.manual_seed(seed)

        # print("SEED", seed)

        if task.task_type == TaskType.REGRESSION:
            self.target_stype = "numerical"
        else:
            self.target_stype = "categorical"

        self.history = train_table.df
        context_df = add_lag_target_features(
            train_table.df,
            train_table.df,
            task,
            num_lags=self.config["num_lags"],
        )
        context = sdm.TableTensor.from_pandas(
            df=context_df,
            stypes={
                task.entity_col: "id",
                task.time_col: "datetime",
                task.target_col: self.target_stype,
                **dict.fromkeys(
                    (
                        f"{task.target_col}_lag_{lag}"
                        for lag in range(1, self.config["num_lags"] + 1)
                    ),
                    self.target_stype,
                ),
            },
        )

        self.expand_query = False
        context_size = self.config["context_size"]
        num_estimators = self.config["num_estimators"]
        generator = torch.Generator().manual_seed(seed)
        # if len(context) > context_size * num_estimators:
        #     perm = context.datetime.view(-1).argsort(descending=True)
        #     context = context[perm]
        #     context = context[:context_size * num_estimators]
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
                context = context.unflatten(0, (num_estimators, context_size))
                num_estimators = None
                self.expand_query = True
        else:
            perm = torch.randperm(len(context), generator=generator)
            context = context[perm[:context_size]]

        self.sampler = get_sampler(db)
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
        print(related_tables.tables.keys())

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

        query_df = add_lag_target_features(
            table.df,
            self.history,
            task,
            num_lags=self.config["num_lags"],
        )
        query = sdm.TableTensor.from_pandas(
            df=query_df,
            stypes={
                task.entity_col: "id",
                task.time_col: "datetime",
                **dict.fromkeys(
                    (
                        f"{task.target_col}_lag_{lag}"
                        for lag in range(1, self.config["num_lags"] + 1)
                    ),
                    self.target_stype,
                ),
            },
        )
        if self.expand_query:
            query = query.expand(self.config["num_estimators"], *query.size())

        outs = []
        for batch in tqdm.tqdm(query.split(10_000, dim=-2)):
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
            if "1" in out.column_names:
                out = out["1"].numerical.squeeze(-1)
            else:
                out = out["True"].numerical.squeeze(-1)
        elif task.task_type == TaskType.REGRESSION:
            out = out["q500"].numerical.squeeze(-1)

        return out.cpu().numpy()


if __name__ == "__main__":
    from relarena.cli import main

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--num-neighbors", type=int, nargs="*")
    parser.add_argument("--num-lags", type=int)
    known_args, unknown_args = parser.parse_known_args()
    NUM_NEIGHBORS = known_args.num_neighbors
    NUM_LAGS = known_args.num_lags


    main(["--model", KumoRelationalModel.name, *unknown_args])
