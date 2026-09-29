# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare CPU and cuGraph sampling on SALT shipping condition."""

import argparse
from typing import cast

import pandas as pd
import relbench
import torch

import sdm


def evaluate(
    data: sdm.RelationalData,
    context: sdm.TableTensor,
    query: sdm.TableTensor,
    num_neighbors: list[int],
    seed: int,
) -> tuple[int, int, torch.Tensor, torch.Tensor]:
    """Return input row counts, predicted classes, and target classes."""
    torch.manual_seed(seed)
    sampler = data.sampler(
        time_columns={
            "salesdocument": "CREATIONTIMESTAMP",
            "salesdocumentitem": "CREATIONTIMESTAMP",
        }
    )
    kwargs = {
        "task_link": {
            "task_column": "SALESDOCUMENT",
            "table": "salesdocument",
            "table_column": "SALESDOCUMENT",
        },
        "num_neighbors": num_neighbors,
        "task_time_column": "CREATIONTIMESTAMP",
        "temporal_strategy": "uniform",
    }
    sampled_context = sampler(
        cast(sdm.TableTensor, context.to(data.device)), **kwargs
    )
    sampled_query = sampler(
        cast(
            sdm.TableTensor,
            query.drop_columns("SHIPPINGCONDITION").to(data.device),
        ),
        **kwargs,
    )
    context_rows = sum(
        len(table) for table in sampled_context.related_tables.tables.values()
    )
    query_rows = sum(
        len(table) for table in sampled_query.related_tables.tables.values()
    )

    torch.manual_seed(seed)
    device = torch.device("cuda")
    model = sdm.models.KumoRelational(task="classification", device=device)
    sampled_context = sampled_context.to(device)
    sampled_query = sampled_query.to(device)
    with torch.amp.autocast("cuda", torch.float16):
        model.fit(
            x=sampled_context.task_table.drop_columns("SHIPPINGCONDITION"),
            y=sampled_context.task_table["SHIPPINGCONDITION"],
            related_tables=sampled_context.related_tables,
        )
        prediction = model.predict(*sampled_query)

    scores, target = sdm.evaluation.to_class_indices(
        prediction,
        cast(sdm.TableTensor, query["SHIPPINGCONDITION"].to(device)),
        missing_score=0,
    )
    return context_rows, query_rows, scores.argmax(-1), target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context-size", type=int, default=2_000)
    parser.add_argument("--query-size", type=int, default=2_048)
    parser.add_argument(
        "--num-neighbors", type=int, nargs="+", default=[-1, -1]
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    task = relbench.load_dataset("rel-salt").load_task("sales-shipcond")
    db = task.get_db(upto_test_timestamp=False)
    data = sdm.RelationalData(
        tables={
            name: sdm.TableTensor.from_pandas(
                df=table.df,
                stypes=sdm.infer_stypes(
                    table.df.head(10_000),
                    overrides={
                        cast(str, table.pkey_col): "id",
                        **dict.fromkeys(table.fkey_col_to_pkey_table, "id"),
                    },
                    text="drop",
                    unsupported="drop",
                ),
            )
            for name, table in db.table_dict.items()
        },
        relationships=[
            {
                "left_table": left_table,
                "left_column": left_column,
                "right_table": right_table,
                "right_column": cast(str, db.table_dict[right_table].pkey_col),
            }
            for left_table, table in db.table_dict.items()
            for left_column, right_table in (
                table.fkey_col_to_pkey_table.items()
            )
        ],
    )

    train = task.get_table("train", mask_input_cols=False).df
    val = task.get_table("val", mask_input_cols=False).df
    test = task.get_table("test", mask_input_cols=False).df
    task_table = sdm.TableTensor.from_pandas(
        df=pd.concat([train, val, test], ignore_index=True),
        stypes={
            "SALESDOCUMENT": "id",
            "CREATIONTIMESTAMP": "datetime",
            "SHIPPINGCONDITION": "categorical",
        },
    )
    context, query = task_table.split([len(train) + len(val), len(test)])
    context = context[torch.randperm(len(context))[: args.context_size]]
    query = query[: args.query_size]

    source_rows = sum(len(table) for table in data.tables.values())
    print(f"Source rows held by the sampler: {source_rows:,}")
    predictions = []
    for name in ("PyG (CPU)", "cuGraph (CUDA)"):
        source = data if name == "PyG (CPU)" else data.to("cuda")
        context_rows, query_rows, predicted, target = evaluate(
            data=source,
            context=context,
            query=query,
            num_neighbors=args.num_neighbors,
            seed=args.seed,
        )
        predictions.append(predicted)
        accuracy = predicted.eq(target).float().mean().item()
        print(
            f"{name}: context rows={context_rows:,} "
            f"({source_rows / context_rows:.1f}x smaller), "
            f"query rows={query_rows:,} "
            f"({source_rows / query_rows:.1f}x smaller), "
            f"accuracy={accuracy:.4f}"
        )
    agreement = predictions[0].eq(predictions[1]).float().mean().item()
    print(f"CPU/CUDA prediction agreement: {agreement:.4f}")


if __name__ == "__main__":
    main()
