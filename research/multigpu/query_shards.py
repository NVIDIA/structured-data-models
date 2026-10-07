"""Assign complete query batches and validate ordered multi-host gathering."""

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def _validate_ids(batch_row_ids: Sequence[Sequence[int]]) -> None:
    flattened = [row for batch in batch_row_ids for row in batch]
    if len(set(flattened)) != len(flattened):
        raise ValueError("Observation IDs must be globally unique")


def plan_batches(
    batch_row_ids: Sequence[Sequence[int]], workers: int
) -> list[list[int]]:
    """Distribute whole original batches without repartitioning their rows.

    Row IDs identify observations, not relational entities. The same entity
    at different timestamps must therefore have different observation IDs.
    Empty worker assignments are retained when workers outnumber batches.
    """
    if workers < 1:
        raise ValueError("At least one worker is required")
    _validate_ids(batch_row_ids)
    return [
        list(range(worker, len(batch_row_ids), workers))
        for worker in range(workers)
    ]


def weighted_plan_batches(
    batch_row_ids: Sequence[Sequence[int]], weights: Sequence[float]
) -> list[list[int]]:
    """Assign whole batches by accumulated row count per measured capacity.

    Weights should come from comparable observed worker throughputs. At each
    step, choose the smallest accumulated rows/weight; ties choose the lower
    worker index. Original batch order and within-batch row order are intact.
    """
    if not weights or any(not math.isfinite(w) or w <= 0 for w in weights):
        raise ValueError("Worker weights must be positive and finite")
    _validate_ids(batch_row_ids)
    assigned: list[list[int]] = [[] for _ in weights]
    rows = [0] * len(weights)
    for index, batch in enumerate(batch_row_ids):
        worker = min(range(len(weights)), key=lambda i: rows[i] / weights[i])
        assigned[worker].append(index)
        rows[worker] += len(batch)
    return assigned


def gather_batches(
    results: Sequence[Mapping[str, Any]],
    expected_row_ids: Sequence[Sequence[int]],
    expected_columns: Sequence[str | int],
    *,
    reject_nonfinite: bool = True,
) -> np.ndarray:
    """Validate complete results and concatenate in original batch order.

    Each result contains ``batch_index``, ``row_ids``, ``predictions`` (a
    two-dimensional NumPy array), and ``columns`` in prediction-column order.
    No row, column, or dtype coercion is performed while gathering. Inputs
    from different workers must use exactly the same dtype and class schema.
    """
    _validate_ids(expected_row_ids)
    ordered: dict[int, np.ndarray] = {}
    dtype = None
    for result in results:
        index = result["batch_index"]
        if isinstance(index, bool) or not isinstance(index, int):
            raise ValueError("Batch index must be an integer")
        if not 0 <= index < len(expected_row_ids):
            raise ValueError(f"Unexpected batch index {index}")
        if index in ordered:
            raise ValueError(f"Duplicate batch index {index}")
        if tuple(result["row_ids"]) != tuple(expected_row_ids[index]):
            raise ValueError(f"Observation IDs differ in batch {index}")
        if tuple(result["columns"]) != tuple(expected_columns):
            raise ValueError(f"Prediction columns differ in batch {index}")
        values = result["predictions"]
        if not isinstance(values, np.ndarray):
            raise ValueError("Predictions must be NumPy arrays")
        expected_shape = (len(expected_row_ids[index]), len(expected_columns))
        if values.shape != expected_shape:
            raise ValueError(
                f"Prediction shape differs in batch {index}: "
                f"{values.shape} versus {expected_shape}"
            )
        if not np.issubdtype(values.dtype, np.number):
            raise ValueError("Predictions must have a numerical dtype")
        if dtype is not None and values.dtype != dtype:
            raise ValueError("Prediction dtypes differ across batches")
        if reject_nonfinite and not np.isfinite(values).all():
            raise ValueError(f"Nonfinite predictions in batch {index}")
        dtype = values.dtype
        ordered[index] = values
    missing = set(range(len(expected_row_ids))) - ordered.keys()
    if missing:
        raise ValueError(f"Missing batches: {sorted(missing)}")
    if not expected_row_ids:
        return np.empty((0, len(expected_columns)), dtype=np.float32)
    return np.concatenate(
        [ordered[index] for index in range(len(expected_row_ids))]
    )
