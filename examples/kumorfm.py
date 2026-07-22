"""Benchmark KumoRFM on complete RelBench entity-task test splits.

Train and validation rows provide the in-context examples. The script samples
their relational neighborhoods, predicts every test row in batches, and
reports the official RelBench metrics. With no dataset filter, it runs every
task in the canonical ``rel-*`` datasets that this benchmark supports.
"""

import argparse
import warnings
from collections.abc import Sequence
from time import perf_counter

import torch
from relbench.base import Dataset, EntityTask
from relbench.datasets import get_dataset, get_dataset_names
from relbench.tasks import get_task, get_task_names
from sdm import (
    RelationalData,
    Relationship,
    Stype,
    StypeLike,
    TableTensor,
    TaskLink,
    infer_stypes,
)
from sdm.models import KumoRFM

INTERFACES = ("forward", "fit-predict")
BENCHMARK_TASK_TYPES = (
    "regression",
    "binary_classification",
    "multiclass_classification",
)


def prediction_to_tensor(
    prediction: TableTensor,
    task_type: str,
) -> torch.Tensor:
    """Convert KumoRFM output into the shape expected by RelBench."""
    columns = list(prediction.columns[Stype.numerical])
    values = prediction.numerical.detach().to(
        device="cpu",
        dtype=torch.float32,
    )

    if task_type == "regression":
        return values[..., columns.index("q500")]

    class_indices = [
        int(column == "True")
        if column in {"False", "True"}
        else int(float(column))
        for column in columns
    ]
    if task_type == "binary_classification":
        return values[..., class_indices.index(1)]

    order = torch.tensor(class_indices).argsort()
    return values.index_select(dim=-1, index=order)


def build_relational_data(db) -> tuple[RelationalData, dict[str, list[str]]]:
    """Convert a RelBench database into SDM relational data."""
    tables: dict[str, TableTensor] = {}
    skipped_columns: dict[str, list[str]] = {}

    for table_name, table in db.table_dict.items():
        id_columns = set(table.fkey_col_to_pkey_table)
        if table.pkey_col is not None:
            id_columns.add(table.pkey_col)

        stypes: dict[str, StypeLike] = {}
        skipped: list[str] = []
        for column in table.df.columns:
            overrides = {column: Stype.id} if column in id_columns else None
            try:
                stypes.update(
                    infer_stypes(table.df[[column]], overrides=overrides)
                )
            except TypeError as error:
                if "Unsupported Arrow type" not in str(error):
                    raise
                if column in id_columns:
                    raise TypeError(
                        f"Cannot skip relational key column {column!r}"
                    ) from error
                skipped.append(column)

        tables[table_name] = TableTensor.from_pandas(
            df=table.df[list(stypes)],
            stypes=stypes,
        )
        if skipped:
            skipped_columns[table_name] = skipped

    relationships = []
    for left_table, table in db.table_dict.items():
        for left_column, right_table in table.fkey_col_to_pkey_table.items():
            right_column = db.table_dict[right_table].pkey_col
            if right_column is None:
                raise ValueError(
                    f"Expected {right_table!r} to have a primary key"
                )
            relationships.append(
                Relationship(
                    left_table=left_table,
                    left_columns=(left_column,),
                    right_table=right_table,
                    right_columns=(right_column,),
                )
            )

    return RelationalData(
        tables=tables,
        relationships=relationships,
    ), skipped_columns


def sample_context(
    df,
    *,
    context_size: int,
    task_type: str,
    target_column: str,
    seed: int,
):
    """Sample context rows while retaining classification coverage."""
    context_size = min(context_size, len(df))
    ordered = df.sample(frac=1.0, random_state=seed)
    if task_type == "regression":
        return ordered.iloc[:context_size].reset_index(drop=True)

    class_rank = ordered.groupby(
        target_column,
        sort=False,
        dropna=False,
    ).cumcount()
    return (
        ordered.assign(__class_rank__=class_rank)
        .sort_values("__class_rank__", kind="stable")
        .iloc[:context_size]
        .drop(columns="__class_rank__")
        .reset_index(drop=True)
    )


def collect_predictions(
    *,
    model: KumoRFM,
    interface: str,
    sampler,
    sampled_context,
    test_table: TableTensor,
    batch_size: int,
    sample_kwargs: dict[str, object],
    task_type: str,
    target_column: str,
    device: torch.device,
    seed: int,
) -> tuple[torch.Tensor, float]:
    """Run one model interface over every test batch."""
    context = sampled_context.to(device)
    context_x = context.task_table.drop_columns(target_column)
    context_y = context.task_table[target_column]
    predictions: list[torch.Tensor] = []
    model_call_runtimes_seconds: list[float] = []
    generator = torch.Generator(device=device).manual_seed(seed)

    try:
        if interface == "fit-predict":
            model.fit(
                x=context_x,
                y=context_y,
                related_tables=context.related_tables,
                generator=generator,
            )

        for test_batch in test_table.split(batch_size):
            query = sampler(test_batch, **sample_kwargs).to(device)
            if device.type == "cuda":
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
            else:
                start_time = perf_counter()
            if interface == "forward":
                prediction = model(
                    x_context=context_x,
                    y_context=context_y,
                    x_query=query.task_table,
                    related_context_tables=context.related_tables,
                    related_query_tables=query.related_tables,
                    generator=generator,
                )
            else:
                prediction = model.predict(
                    x=query.task_table,
                    related_tables=query.related_tables,
                )
            if device.type == "cuda":
                end_event.record()
            else:
                model_call_runtime_seconds = perf_counter() - start_time
            predictions.append(prediction_to_tensor(prediction, task_type))
            if device.type == "cuda":
                model_call_runtime_seconds = (
                    start_event.elapsed_time(end_event) / 1_000
                )
            model_call_runtimes_seconds.append(model_call_runtime_seconds)
    finally:
        model.clear()

    return (
        torch.cat(predictions, dim=0),
        sum(model_call_runtimes_seconds) / len(model_call_runtimes_seconds),
    )


def run_benchmark(
    *,
    dataset: str,
    task_name: str,
    context_size: int,
    batch_size: int,
    num_neighbors: Sequence[int],
    interfaces: Sequence[str],
    device: torch.device,
    seed: int,
) -> dict[str, object]:
    """Evaluate KumoRFM over a complete supported RelBench test split."""
    import pandas as pd

    # AutoCompleteTask removes target and leakage columns while it is created,
    # so construct the task before materializing its database.
    task = get_task(dataset, task_name, download=True)
    if not isinstance(task, EntityTask):
        return {
            "dataset": dataset,
            "task": task_name,
            "status": "skipped",
            "reason": "This benchmark only supports RelBench entity tasks",
        }

    task_type = task.task_type.value
    if task_type not in BENCHMARK_TASK_TYPES:
        return {
            "dataset": dataset,
            "task": task_name,
            "status": "skipped",
            "reason": (
                f"This benchmark does not support task type {task_type!r}"
            ),
        }

    num_classes = None
    if task_type == "binary_classification":
        num_classes = 2
    elif task_type == "multiclass_classification":
        num_classes = task.num_classes
        if num_classes is None:
            raise ValueError("Multiclass task does not declare its classes")
    if num_classes is not None and num_classes > 10:
        return {
            "dataset": dataset,
            "task": task_name,
            "status": "skipped",
            "reason": (
                f"This benchmark supports at most 10 classes; task has "
                f"{num_classes}"
            ),
        }

    db = task.dataset.get_db(upto_test_timestamp=False)
    data, skipped_columns = build_relational_data(db)
    if skipped_columns:
        skipped = [
            f"{table}.{column}"
            for table, columns in skipped_columns.items()
            for column in columns
        ]
        warnings.warn(
            "Skipping columns with unsupported Arrow types: "
            + ", ".join(skipped),
            stacklevel=2,
        )

    sampler = data.sampler(
        time_columns={
            table_name: table.time_col
            for table_name, table in db.table_dict.items()
            if table.time_col is not None
        },
    )

    context_df = pd.concat(
        [
            task.get_table("train", mask_input_cols=False).df,
            task.get_table("val", mask_input_cols=False).df,
        ],
        ignore_index=True,
    ).dropna(subset=[task.target_col])
    context_df = sample_context(
        context_df,
        context_size=context_size,
        task_type=task_type,
        target_column=task.target_col,
        seed=seed,
    )
    if (
        num_classes is not None
        and context_df[task.target_col].nunique() != num_classes
    ):
        raise ValueError(
            "The sampled context does not contain every target class; "
            "increase '--context-size'."
        )

    context_table = TableTensor.from_pandas(
        df=context_df,
        stypes={
            task.entity_col: Stype.id,
            task.time_col: Stype.datetime,
            task.target_col: Stype.numerical
            if task_type == "regression"
            else Stype.categorical,
        },
    )
    target_table = task.get_table("test", mask_input_cols=False)
    test_table = TableTensor.from_pandas(
        df=target_table.df[[task.entity_col, task.time_col]],
        stypes={
            task.entity_col: Stype.id,
            task.time_col: Stype.datetime,
        },
    )

    entity_primary_key = db.table_dict[task.entity_table].pkey_col
    if entity_primary_key is None:
        raise ValueError(
            f"Expected {task.entity_table!r} to have a primary key"
        )
    sample_kwargs: dict[str, object] = {
        "task_link": TaskLink(
            task_columns=(task.entity_col,),
            table=task.entity_table,
            table_columns=(entity_primary_key,),
        ),
        "num_neighbors": num_neighbors,
        "task_time_column": task.time_col,
    }
    sampled_context = sampler(context_table, **sample_kwargs)

    interface_results: dict[str, object] = {}
    for interface in interfaces:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        prediction, average_model_call_runtime_seconds = collect_predictions(
            model=KumoRFM(device=device),
            interface=interface,
            sampler=sampler,
            sampled_context=sampled_context,
            test_table=test_table,
            batch_size=batch_size,
            sample_kwargs=sample_kwargs,
            task_type=task_type,
            target_column=task.target_col,
            device=device,
            seed=seed,
        )

        metrics = task.evaluate(
            prediction.numpy(),
            target_table=target_table,
        )
        interface_results[interface] = {
            "metrics": {name: float(value) for name, value in metrics.items()},
            "average_batch_model_call_runtime_seconds": (
                average_model_call_runtime_seconds
            ),
            "peak_cuda_memory_allocated_bytes": (
                torch.cuda.max_memory_allocated(device)
                if device.type == "cuda"
                else None
            ),
        }

    return {
        "dataset": dataset,
        "task": task_name,
        "status": "completed",
        "task_type": task_type,
        "context_rows": len(context_table),
        "test_rows": len(test_table),
        "interfaces": interface_results,
        "skipped_columns": skipped_columns,
    }


def print_result(result) -> None:
    """Print one benchmark result."""
    lines = [
        f"{result['dataset']}/{result['task']} [{result['status']}]",
    ]
    if result["status"] == "completed":
        for interface, values in result["interfaces"].items():
            metrics = ", ".join(
                f"{name}={value:.6g}"
                for name, value in values["metrics"].items()
            )
            lines.append(f"  {interface}: {metrics}")
            performance = (
                f"    average_batch_model_call_runtime="
                f"{values['average_batch_model_call_runtime_seconds']:.3f}s"
            )
            peak_memory = values["peak_cuda_memory_allocated_bytes"]
            if peak_memory is not None:
                performance += (
                    f", peak_cuda_memory_allocated={peak_memory} bytes"
                )
            lines.append(performance)
    else:
        lines.append(f"  {result['reason']}")
    print("\n".join(lines), flush=True)  # noqa: T201


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate KumoRFM on complete RelBench test splits. With no "
            "filters, run every task in the standard datasets; with "
            "--dataset, run every task in that dataset; with --dataset and "
            "--task, run one task."
        )
    )
    parser.add_argument(
        "--dataset",
        help="RelBench dataset to run; omit to run all standard datasets",
    )
    parser.add_argument(
        "--task",
        help="Task to run from --dataset; omit to run every selected task",
    )
    parser.add_argument("--context-size", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=1_000)
    parser.add_argument(
        "--num-neighbors",
        type=int,
        nargs="+",
        default=[16, 16],
    )
    parser.add_argument(
        "--interface",
        choices=INTERFACES,
        default="fit-predict",
        help="SDM model interface to benchmark (default: fit-predict)",
    )
    parser.add_argument("--device")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.task and not args.dataset:
        parser.error("'--task' requires '--dataset'")
    return args


def main() -> None:
    """Run the command-line benchmark."""
    args = _parse_args()
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    available_datasets = sorted(get_dataset_names())
    if args.dataset and args.dataset not in available_datasets:
        raise ValueError(
            f"Unknown RelBench dataset {args.dataset!r}. Available datasets: "
            f"{', '.join(available_datasets)}"
        )
    datasets = (
        [args.dataset]
        if args.dataset
        else [
            name
            for name in available_datasets
            if name.startswith("rel-") and name != "rel-salt"
        ]
    )
    if args.task and args.task not in get_task_names(args.dataset):
        raise ValueError(
            f"Unknown task {args.task!r} for dataset {args.dataset!r}. "
            f"Available tasks: {', '.join(get_task_names(args.dataset))}"
        )
    benchmark_tasks = [
        (dataset, task_name)
        for dataset in datasets
        for task_name in (
            [args.task] if args.task else sorted(get_task_names(dataset))
        )
    ]
    if not benchmark_tasks:
        raise ValueError(
            f"No registered RelBench tasks for dataset {args.dataset!r}"
        )

    for dataset, task_name in benchmark_tasks:
        result = run_benchmark(
            dataset=dataset,
            task_name=task_name,
            context_size=args.context_size,
            batch_size=args.batch_size,
            num_neighbors=args.num_neighbors,
            interfaces=(args.interface,),
            device=device,
            seed=args.seed,
        )
        if args.task and result["status"] == "skipped":
            raise ValueError(result["reason"])
        print_result(result)
        # RelBench caches complete tasks and databases in memory.
        Dataset.get_db.cache_clear()
        get_task.cache_clear()
        get_dataset.cache_clear()


if __name__ == "__main__":
    main()
