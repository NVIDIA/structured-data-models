from __future__ import annotations

import argparse
import json
import warnings
from collections.abc import Collection, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal, cast

import pandas as pd
import torch
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

if TYPE_CHECKING:
    from relbench.base import Database, EntityTask
    from sdm.relational import RelationalSampler, RelationalSamplerOutput

Interface = Literal["forward", "fit-predict"]
TaskKind = Literal[
    "regression",
    "binary_classification",
    "multiclass_classification",
]

MAX_CLASSES = 10
INTERFACES: tuple[Interface, ...] = ("forward", "fit-predict")
TASK_KINDS: tuple[TaskKind, ...] = (
    "regression",
    "binary_classification",
    "multiclass_classification",
)


def _class_index(value: object) -> int:
    text = str(value)
    if text.casefold() == "false":
        return 0
    if text.casefold() == "true":
        return 1

    try:
        number = float(text)
    except ValueError as error:
        raise ValueError(
            f"Expected a numerical class label, got {value!r}"
        ) from error

    index = int(number)
    if not number.is_integer() or index < 0:
        raise ValueError(
            f"Expected a non-negative integer class label, got {value!r}"
        )
    return index


def prediction_to_tensor(
    prediction: TableTensor,
    task_kind: TaskKind,
    *,
    num_classes: int | None = None,
) -> torch.Tensor:
    """Convert KumoRFM output into the shape expected by RelBench."""
    columns = prediction.columns[Stype.numerical]
    values = prediction.numerical.detach().to(
        device="cpu",
        dtype=torch.float32,
    )

    if task_kind == "regression":
        if "q500" not in columns:
            raise ValueError(
                "Expected regression output to contain median quantile "
                f"'q500', got {list(columns)}"
            )
        return values[..., columns.index("q500")]

    indices = [_class_index(column) for column in columns]
    if task_kind == "binary_classification":
        if len(indices) != 2 or set(indices) != {0, 1}:
            raise ValueError(
                "Expected binary prediction columns for classes 0 and 1, "
                f"got {list(columns)}"
            )
        return values[..., indices.index(1)]

    if task_kind == "multiclass_classification":
        if num_classes is None:
            raise ValueError("Expected 'num_classes' for multiclass output")
        if len(indices) != num_classes or set(indices) != set(
            range(num_classes)
        ):
            raise ValueError(
                "Expected multiclass prediction columns for every class in "
                f"[0, {num_classes}), got {list(columns)}"
            )
        output = values.new_empty((*values.shape[:-1], num_classes))
        index = torch.tensor(indices, dtype=torch.long)
        output.index_copy_(dim=-1, index=index, source=values)
        return output

    raise ValueError(f"Unsupported task kind {task_kind!r}")


def _infer_table_stypes(
    df: pd.DataFrame,
    *,
    id_columns: Collection[str],
) -> tuple[dict[str, StypeLike], list[str]]:
    stypes: dict[str, StypeLike] = {}
    skipped: list[str] = []
    id_columns = set(id_columns)

    for column in df.columns:
        overrides = {column: Stype.id} if column in id_columns else None
        try:
            stypes.update(infer_stypes(df[[column]], overrides=overrides))
        except TypeError as error:
            if "Unsupported Arrow type" not in str(error):
                raise
            if column in id_columns:
                raise TypeError(
                    f"Cannot skip relational key column {column!r}"
                ) from error
            skipped.append(column)

    return stypes, skipped


def _primary_key(db: Database, table_name: str) -> str:
    primary_key = db.table_dict[table_name].pkey_col
    if primary_key is None:
        raise ValueError(
            f"Expected referenced table {table_name!r} to have a primary key"
        )
    return primary_key


def build_relational_data(
    db: Database,
) -> tuple[RelationalData, dict[str, list[str]]]:
    """Tensorize a RelBench database while preserving relational keys."""
    tables: dict[str, TableTensor] = {}
    skipped_columns: dict[str, list[str]] = {}

    for table_name, table in db.table_dict.items():
        id_columns = set(table.fkey_col_to_pkey_table)
        if table.pkey_col is not None:
            id_columns.add(table.pkey_col)
        stypes, skipped = _infer_table_stypes(
            table.df,
            id_columns=id_columns,
        )
        tables[table_name] = TableTensor.from_pandas(
            df=table.df[list(stypes)],
            stypes=stypes,
        )
        if skipped:
            skipped_columns[table_name] = skipped

    relationships = [
        Relationship(
            left_table=left_table,
            left_columns=(left_column,),
            right_table=right_table,
            right_columns=(_primary_key(db, right_table),),
        )
        for left_table, table in db.table_dict.items()
        for left_column, right_table in table.fkey_col_to_pkey_table.items()
    ]
    return RelationalData(
        tables=tables,
        relationships=relationships,
    ), skipped_columns


def sample_context(
    df: pd.DataFrame,
    *,
    context_size: int,
    task_kind: TaskKind,
    target_column: str,
    seed: int,
) -> pd.DataFrame:
    """Sample context rows while prioritizing classification coverage."""
    context_size = min(context_size, len(df))
    ordered = df.sample(frac=1.0, random_state=seed)
    if task_kind == "regression":
        return ordered.iloc[:context_size].reset_index(drop=True)

    ranks = ordered.groupby(
        target_column,
        sort=False,
        dropna=False,
    ).cumcount()
    return (
        ordered.assign(__class_rank__=ranks)
        .sort_values("__class_rank__", kind="stable")
        .iloc[:context_size]
        .drop(columns="__class_rank__")
        .reset_index(drop=True)
    )


def _validate_context_classes(
    context_df: pd.DataFrame,
    *,
    task_kind: TaskKind,
    target_column: str,
    num_classes: int | None,
) -> int | None:
    expected = _validate_num_classes(task_kind, num_classes)
    if expected is None:
        return None

    classes = {
        _class_index(value)
        for value in context_df[target_column].dropna().unique()
    }
    if classes != set(range(expected)):
        raise ValueError(
            f"Expected context classes {list(range(expected))}, got "
            f"{sorted(classes)}. Increase '--context-size'."
        )
    return expected


def _validate_num_classes(
    task_kind: TaskKind,
    num_classes: int | None,
) -> int | None:
    if task_kind == "regression":
        return None

    expected = 2 if task_kind == "binary_classification" else num_classes
    if expected is None:
        raise ValueError("Expected 'num_classes' for multiclass task")
    if expected < 2:
        raise ValueError(
            "Expected a classification task to contain at least two classes"
        )
    if expected > MAX_CLASSES:
        raise NotImplementedError(
            f"KumoRFM supports at most {MAX_CLASSES} classes in this "
            f"benchmark, but the task requires {expected}"
        )
    return expected


def _make_generator(device: torch.device, seed: int) -> torch.Generator:
    return torch.Generator(device=device).manual_seed(seed)


def collect_predictions(
    *,
    model: KumoRFM,
    interface: Interface,
    sampler: RelationalSampler,
    sampled_context: RelationalSamplerOutput,
    test_table: TableTensor,
    batch_size: int,
    sample_kwargs: Mapping[str, Any],
    task_kind: TaskKind,
    target_column: str,
    num_classes: int | None,
    device: torch.device,
    seed: int,
) -> torch.Tensor:
    """Run one model interface over every test batch."""
    if interface not in INTERFACES:
        raise ValueError(f"Unsupported model interface {interface!r}")

    context = sampled_context.to(device)
    context_x = context.task_table.drop_columns(target_column)
    context_y = context.task_table[target_column]
    predictions: list[torch.Tensor] = []

    try:
        if interface == "fit-predict":
            model.fit(
                x=context_x,
                y=context_y,
                related_tables=context.related_tables,
                generator=_make_generator(device, seed),
            )

        for test_batch in test_table.split(batch_size):
            query = sampler(test_batch, **sample_kwargs).to(device)
            if interface == "forward":
                prediction = model(
                    x_context=context_x,
                    y_context=context_y,
                    x_query=query.task_table,
                    related_context_tables=context.related_tables,
                    related_query_tables=query.related_tables,
                    generator=_make_generator(device, seed),
                )
            else:
                prediction = model.predict(
                    x=query.task_table,
                    related_tables=query.related_tables,
                )
            predictions.append(
                prediction_to_tensor(
                    prediction,
                    task_kind,
                    num_classes=num_classes,
                )
            )
    finally:
        model.clear()

    prediction = torch.cat(predictions, dim=0)
    if len(prediction) != len(test_table):
        raise RuntimeError(
            f"Expected {len(test_table)} predictions, got {len(prediction)}"
        )
    return prediction


def _task_kind(task: EntityTask) -> TaskKind:
    value = task.task_type.value
    if value not in TASK_KINDS:
        raise NotImplementedError(
            f"KumoRFM does not support RelBench task type {value!r} in this "
            "benchmark"
        )
    return value


def _resolve_device(value: str | None) -> torch.device:
    if value is None:
        value = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def run_benchmark(
    *,
    dataset: str,
    task_name: str,
    context_size: int,
    batch_size: int,
    num_neighbors: Sequence[int],
    interfaces: Sequence[Interface],
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    """Evaluate KumoRFM over a complete supported RelBench test split."""
    if context_size <= 0:
        raise ValueError("'context_size' must be positive")
    if batch_size <= 0:
        raise ValueError("'batch_size' must be positive")
    if not num_neighbors or min(num_neighbors) <= 0:
        raise ValueError("'num_neighbors' values must be positive")
    if not interfaces:
        raise ValueError("At least one model interface is required")
    invalid_interfaces = set(interfaces) - set(INTERFACES)
    if invalid_interfaces:
        raise ValueError(
            f"Unsupported model interfaces: {sorted(invalid_interfaces)}"
        )
    if not 0 <= seed < 2**32:
        raise ValueError("'seed' must be in the range [0, 2**32)")

    from relbench.base import EntityTask
    from relbench.tasks import get_task

    # AutoCompleteTask removes target and leakage columns while it is created,
    # so construct the task before materializing its database.
    task = get_task(dataset, task_name, download=True)
    if not isinstance(task, EntityTask):
        raise NotImplementedError(
            "KumoRFM's public interfaces support RelBench entity tasks, not "
            f"{task.__class__.__name__}"
        )
    task_kind = _task_kind(task)
    declared_num_classes = _validate_num_classes(
        task_kind,
        getattr(task, "num_classes", None),
    )

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
    )
    context_df = context_df.dropna(subset=[task.target_col])
    if len(context_df) == 0:
        raise ValueError("Expected the context set to contain labeled rows")
    context_df = sample_context(
        context_df,
        context_size=context_size,
        task_kind=task_kind,
        target_column=task.target_col,
        seed=seed,
    )
    num_classes = _validate_context_classes(
        context_df,
        task_kind=task_kind,
        target_column=task.target_col,
        num_classes=declared_num_classes,
    )

    context_table = TableTensor.from_pandas(
        df=context_df,
        stypes={
            task.entity_col: Stype.id,
            task.time_col: Stype.datetime,
            task.target_col: Stype.numerical
            if task_kind == "regression"
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
    if len(test_table) == 0:
        raise ValueError("Expected the test set to contain at least one row")

    sample_kwargs: dict[str, Any] = {
        "task_link": TaskLink(
            task_columns=(task.entity_col,),
            table=task.entity_table,
            table_columns=(_primary_key(db, task.entity_table),),
        ),
        "num_neighbors": num_neighbors,
        "task_time_column": task.time_col,
    }
    sampled_context = sampler(context_table, **sample_kwargs)

    interface_results: dict[str, Any] = {}
    for interface in interfaces:
        prediction = collect_predictions(
            model=KumoRFM(device=device),
            interface=interface,
            sampler=sampler,
            sampled_context=sampled_context,
            test_table=test_table,
            batch_size=batch_size,
            sample_kwargs=sample_kwargs,
            task_kind=task_kind,
            target_column=task.target_col,
            num_classes=num_classes,
            device=device,
            seed=seed,
        )
        metrics = task.evaluate(
            prediction.numpy(),
            target_table=target_table,
        )
        interface_results[interface] = {
            "metrics": {name: float(value) for name, value in metrics.items()},
        }

    return {
        "dataset": dataset,
        "task": task_name,
        "task_type": task_kind,
        "context_rows": len(context_table),
        "test_rows": len(test_table),
        "interfaces": interface_results,
        "skipped_columns": skipped_columns,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate KumoRFM over an entire RelBench test split"
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--task", required=True)
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
        choices=(*INTERFACES, "both"),
        default="fit-predict",
    )
    parser.add_argument("--device")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    """Run the command-line benchmark."""
    args = _parse_args()
    interfaces: tuple[Interface, ...]
    if args.interface == "both":
        interfaces = ("forward", "fit-predict")
    else:
        interfaces = (cast(Interface, args.interface),)
    result = run_benchmark(
        dataset=args.dataset,
        task_name=args.task,
        context_size=args.context_size,
        batch_size=args.batch_size,
        num_neighbors=args.num_neighbors,
        interfaces=interfaces,
        device=_resolve_device(args.device),
        seed=args.seed,
    )
    print(json.dumps(result, indent=2, sort_keys=True))  # noqa: T201


if __name__ == "__main__":
    main()
