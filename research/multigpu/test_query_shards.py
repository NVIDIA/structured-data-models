# ruff: noqa: D103, TID253
"""Observation and class identity checks for whole-batch sharding."""

import numpy as np
import pytest
from research.multigpu.query_shards import gather_batches, plan_batches


def _results() -> list[dict]:
    return [
        {
            "batch_index": 1,
            "row_ids": [19],
            "columns": ["1", "0"],
            "predictions": np.array([[0.6, 0.4]], dtype=np.float32),
        },
        {
            "batch_index": 0,
            "row_ids": [7, 12],
            "columns": ["1", "0"],
            "predictions": np.array(
                [[0.1, 0.9], [0.3, 0.7]], dtype=np.float32
            ),
        },
    ]


def test_plan_preserves_complete_unequal_batches() -> None:
    assert plan_batches([[7, 12], [19], [25, 29, 31]], 2) == [[0, 2], [1]]
    assert plan_batches([[7, 12], [19]], 4) == [[0], [1], [], []]


def test_gather_preserves_global_order_and_class_schema() -> None:
    out = gather_batches(_results(), [[7, 12], [19]], ["1", "0"])
    expected = np.array([[0.1, 0.9], [0.3, 0.7], [0.6, 0.4]], dtype=np.float32)
    assert out.dtype == np.float32
    assert out.tobytes() == expected.tobytes()


@pytest.mark.parametrize("workers", [0, -1])
def test_invalid_worker_count(workers: int) -> None:
    with pytest.raises(ValueError, match="worker"):
        plan_batches([[0]], workers)


def test_reject_duplicate_observation_ids() -> None:
    with pytest.raises(ValueError, match="globally unique"):
        plan_batches([[7], [7]], 2)


@pytest.mark.parametrize(
    "kind",
    [
        "duplicate",
        "missing",
        "out_of_range",
        "negative",
        "row_ids",
        "columns",
        "row_count",
        "width",
        "dtype",
        "nonfinite",
    ],
)
def test_reject_corrupted_worker_result(kind: str) -> None:
    results = _results()
    if kind == "duplicate":
        results.append(results[0])
    elif kind == "missing":
        results.pop()
    elif kind == "out_of_range":
        results[0]["batch_index"] = 2
    elif kind == "negative":
        results[0]["batch_index"] = -1
    elif kind == "row_ids":
        results[1]["row_ids"] = [12, 7]
    elif kind == "columns":
        results[0]["columns"] = ["0", "1"]
    elif kind == "row_count":
        results[0]["predictions"] = np.ones((2, 2), dtype=np.float32)
    elif kind == "width":
        results[0]["predictions"] = np.ones((1, 3), dtype=np.float32)
    elif kind == "dtype":
        results[0]["predictions"] = results[0]["predictions"].astype(
            np.float64
        )
    else:
        results[0]["predictions"][0, 0] = np.nan
    messages = {
        "duplicate": "Duplicate",
        "missing": "Missing",
        "out_of_range": "Unexpected",
        "negative": "Unexpected",
        "row_ids": "Observation IDs",
        "columns": "columns differ",
        "row_count": "shape differs",
        "width": "shape differs",
        "dtype": "dtypes differ",
        "nonfinite": "Nonfinite",
    }
    with pytest.raises(ValueError, match=messages[kind]):
        gather_batches(results, [[7, 12], [19]], ["1", "0"])


def test_empty_plan_and_gather() -> None:
    assert plan_batches([], 3) == [[], [], []]
    assert gather_batches([], [], ["score"]).shape == (0, 1)


def test_optional_nonfinite_passthrough_for_failure_reporting() -> None:
    results = _results()
    results[0]["predictions"][0, 0] = np.nan
    out = gather_batches(
        results, [[7, 12], [19]], ["1", "0"], reject_nonfinite=False
    )
    assert np.isnan(out[-1, 0])
