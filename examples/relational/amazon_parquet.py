# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run KumoRelational on rel-amazon with Parquet-backed sampling."""

import argparse
import resource
from typing import cast

import pandas as pd
import polars as pl
import pyarrow.parquet as pq
import relbench
import torch

import sdm


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context-size", type=int, default=512)
    parser.add_argument("--query-size", type=int, default=64)
    parser.add_argument("--num-neighbors", type=int, nargs="+", default=[8, 8])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    dataset = relbench.load_dataset("rel-amazon")
    task = dataset.load_task("user-churn")
    tables = {
        name: dataset.db_dir / f"{name}.parquet"
        for name in dataset.manifest.tables
    }

    stypes = {}
    for name, spec in dataset.manifest.tables.items():
        sample = (
            pl.scan_parquet(tables[name])
            .head(10_000)
            .collect(engine="streaming")
            .to_pandas()
        )
        keys = dict.fromkeys(spec.fkeys, "id")
        if spec.pkey is not None:
            keys[spec.pkey] = "id"
        stypes[name] = sdm.infer_stypes(
            sample,
            overrides=keys,
            text="drop",
            unsupported="drop",
        )

    relationships = [
        {
            "left_table": name,
            "left_column": column,
            "right_table": other,
            "right_column": cast(str, dataset.manifest.tables[other].pkey),
        }
        for name, spec in dataset.manifest.tables.items()
        for column, other in spec.fkeys.items()
    ]
    time_columns = {
        name: spec.time_col
        for name, spec in dataset.manifest.tables.items()
        if spec.time_col is not None
    }
    sampler = sdm.ParquetRelationalSampler(
        tables=tables,
        stypes=stypes,
        relationships=relationships,
        time_columns=time_columns,
    )

    train = task.get_table("train", mask_input_cols=False).df
    val = task.get_table("val", mask_input_cols=False).df
    test = task.get_table("test", mask_input_cols=False).df
    task_table = sdm.TableTensor.from_pandas(
        df=pd.concat([train, val, test], ignore_index=True),
        stypes={
            task.entity_col: "id",
            task.time_col: "datetime",
            task.target_col: "categorical",
        },
    )
    context, query = task_table.split([len(train) + len(val), len(test)])
    context = context[torch.randperm(len(context))[: args.context_size]]
    query = query[: args.query_size]

    task_link = {
        "task_column": task.entity_col,
        "table": task.entity_table,
        "table_column": cast(
            str, dataset.manifest.tables[task.entity_table].pkey
        ),
    }
    sampled_context = sampler(
        task_table=context,
        task_link=task_link,
        num_neighbors=args.num_neighbors,
        task_time_column=task.time_col,
    )
    sampled_query = sampler(
        task_table=query.drop_columns(task.target_col),
        task_link=task_link,
        num_neighbors=args.num_neighbors,
        task_time_column=task.time_col,
    )

    source_rows = sum(
        pq.ParquetFile(path).metadata.num_rows for path in tables.values()
    )
    context_rows = sum(
        len(table) for table in sampled_context.related_tables.tables.values()
    )
    query_rows = sum(
        len(table) for table in sampled_query.related_tables.tables.values()
    )
    print(f"Source rows on disk: {source_rows:,}")
    print(f"Context related rows: {context_rows:,}")
    print(f"Query related rows: {query_rows:,}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = sdm.models.KumoRelational(task="classification", device=device)
    sampled_context = sampled_context.to(device)
    sampled_query = sampled_query.to(device)
    with torch.amp.autocast(device.type, enabled=device.type == "cuda"):
        model.fit(
            x=sampled_context.task_table.drop_columns(task.target_col),
            y=sampled_context.task_table[task.target_col],
            related_tables=sampled_context.related_tables,
        )
        prediction = model.predict(*sampled_query)

    scores, target = sdm.evaluation.to_class_indices(
        prediction, cast(sdm.TableTensor, query[task.target_col].to(device))
    )
    accuracy = (scores.argmax(-1) == target).float().mean()
    print(f"Query accuracy: {accuracy:.4f}")
    peak_gib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    print(f"Peak process RAM: {peak_gib:.2f} GiB")


if __name__ == "__main__":
    main()
