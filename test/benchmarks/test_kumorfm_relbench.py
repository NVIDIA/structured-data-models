from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from runpy import run_path
from types import SimpleNamespace
from typing import Any, cast

import pandas as pd
import pytest
import torch
from sdm import Stype, TableTensor

benchmark = SimpleNamespace(
    **run_path(
        str(
            Path(__file__).parents[2]
            / "benchmarks"
            / "kumorfm"
            / "relbench_benchmark.py"
        )
    )
)


@pytest.mark.parametrize(
    ("columns", "values", "task_kind", "num_classes", "expected"),
    [
        (
            ("q001", "q500", "q999"),
            [[-1.0, 2.0, 4.0]],
            "regression",
            None,
            [2.0],
        ),
        (
            ("1", "0"),
            [[0.8, 0.2]],
            "binary_classification",
            None,
            [0.8],
        ),
    ],
)
def test_prediction_to_tensor(
    columns: tuple[str, ...],
    values: list[list[float]],
    task_kind: str,
    num_classes: int | None,
    expected: list[float],
) -> None:
    prediction = TableTensor(
        columns={Stype.numerical: columns},
        numerical=torch.tensor(values),
    )

    actual = benchmark.prediction_to_tensor(
        prediction,
        task_kind,
        num_classes=num_classes,
    )

    torch.testing.assert_close(actual, torch.tensor(expected))


def test_prediction_to_tensor_reorders_multiclass_columns() -> None:
    prediction = TableTensor(
        columns={Stype.numerical: ("2", "0", "1")},
        numerical=torch.tensor([[0.2, 0.3, 0.5]]),
    )

    actual = benchmark.prediction_to_tensor(
        prediction,
        "multiclass_classification",
        num_classes=3,
    )

    torch.testing.assert_close(actual, torch.tensor([[0.3, 0.5, 0.2]]))


@dataclass
class _Table:
    df: pd.DataFrame
    pkey_col: str | None
    fkey_col_to_pkey_table: dict[str, str]


@dataclass
class _Database:
    table_dict: dict[str, _Table]


def test_build_relational_data_preserves_keys_and_skips_lists() -> None:
    database = _Database(
        table_dict={
            "users": _Table(
                df=pd.DataFrame(
                    {
                        "primary": [10, 20],
                        "unsupported": [[1], [2]],
                    }
                ),
                pkey_col="primary",
                fkey_col_to_pkey_table={},
            ),
            "events": _Table(
                df=pd.DataFrame(
                    {
                        "event": [100, 200],
                        "owner": [10, 20],
                        "value": [1.0, 2.0],
                    }
                ),
                pkey_col="event",
                fkey_col_to_pkey_table={"owner": "users"},
            ),
        }
    )

    data, skipped = benchmark.build_relational_data(cast(Any, database))

    assert data.tables["users"].stype("primary") == Stype.id
    assert data.tables["events"].stype("event") == Stype.id
    assert data.tables["events"].stype("owner") == Stype.id
    assert data.tables["events"].stype("value") == Stype.numerical
    assert skipped == {"users": ["unsupported"]}
    assert len(data.relationships) == 1


def test_sample_context_keeps_each_class() -> None:
    source = pd.DataFrame(
        {
            "entity": range(8),
            "target": [0, 0, 0, 0, 0, 0, 1, 1],
        }
    )

    context = benchmark.sample_context(
        source,
        context_size=2,
        task_kind="binary_classification",
        target_column="target",
        seed=0,
    )

    assert set(context["target"]) == {0, 1}


def test_validate_context_rejects_missing_or_too_many_classes() -> None:
    with pytest.raises(ValueError, match="Increase '--context-size'"):
        benchmark._validate_context_classes(
            pd.DataFrame({"target": [0, 0]}),
            task_kind="binary_classification",
            target_column="target",
            num_classes=None,
        )

    with pytest.raises(NotImplementedError, match="at most 10 classes"):
        benchmark._validate_context_classes(
            pd.DataFrame({"target": range(11)}),
            task_kind="multiclass_classification",
            target_column="target",
            num_classes=11,
        )


class _Sample:
    def __init__(self, task_table: TableTensor) -> None:
        self.task_table = task_table
        self.related_tables = object()

    def to(self, device: torch.device) -> _Sample:
        return self


class _Sampler:
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def __call__(
        self,
        task_table: TableTensor,
        **kwargs: object,
    ) -> _Sample:
        self.batch_sizes.append(len(task_table))
        return _Sample(task_table)


class _Model:
    def __init__(self) -> None:
        self.forward_calls = 0
        self.fit_calls = 0
        self.predict_calls = 0
        self.clear_calls = 0

    @staticmethod
    def _prediction(rows: int) -> TableTensor:
        return TableTensor(
            columns={Stype.numerical: ("0", "1")},
            numerical=torch.tensor([[0.25, 0.75]]).repeat(rows, 1),
        )

    def __call__(self, **kwargs: Any) -> TableTensor:
        self.forward_calls += 1
        return self._prediction(len(kwargs["x_query"]))

    def fit(self, **kwargs: Any) -> None:
        self.fit_calls += 1

    def predict(self, **kwargs: Any) -> TableTensor:
        self.predict_calls += 1
        return self._prediction(len(kwargs["x"]))

    def clear(self) -> None:
        self.clear_calls += 1


class _InvalidModel(_Model):
    @staticmethod
    def _prediction(rows: int) -> TableTensor:
        return TableTensor(
            columns={Stype.numerical: ("0",)},
            numerical=torch.ones(rows, 1),
        )


@pytest.mark.parametrize("interface", ["forward", "fit-predict"])
def test_collect_predictions_covers_every_test_row(interface: str) -> None:
    model = _Model()
    sampler = _Sampler()
    context = TableTensor.from_columns(
        {"entity": [0, 1], "target": [False, True]},
        stypes={"entity": Stype.id, "target": Stype.categorical},
    )
    test_table = TableTensor.from_columns(
        {"entity": [2, 3, 4, 5, 6]},
        stypes={"entity": Stype.id},
    )

    prediction = benchmark.collect_predictions(
        model=cast(Any, model),
        interface=interface,
        sampler=cast(Any, sampler),
        sampled_context=cast(Any, _Sample(context)),
        test_table=test_table,
        batch_size=2,
        sample_kwargs={},
        task_kind="binary_classification",
        target_column="target",
        num_classes=2,
        device=torch.device("cpu"),
        seed=0,
    )

    torch.testing.assert_close(prediction, torch.full((5,), 0.75))
    assert sampler.batch_sizes == [2, 2, 1]
    assert model.forward_calls == (3 if interface == "forward" else 0)
    assert model.fit_calls == (1 if interface == "fit-predict" else 0)
    assert model.predict_calls == (3 if interface == "fit-predict" else 0)
    assert model.clear_calls == 1


def test_collect_predictions_clears_cache_after_failure() -> None:
    model = _InvalidModel()
    context = TableTensor.from_columns(
        {"entity": [0, 1], "target": [False, True]},
        stypes={"entity": Stype.id, "target": Stype.categorical},
    )
    test_table = TableTensor.from_columns(
        {"entity": [2]},
        stypes={"entity": Stype.id},
    )

    with pytest.raises(ValueError, match="binary prediction columns"):
        benchmark.collect_predictions(
            model=cast(Any, model),
            interface="fit-predict",
            sampler=cast(Any, _Sampler()),
            sampled_context=cast(Any, _Sample(context)),
            test_table=test_table,
            batch_size=1,
            sample_kwargs={},
            task_kind="binary_classification",
            target_column="target",
            num_classes=2,
            device=torch.device("cpu"),
            seed=0,
        )

    assert model.clear_calls == 1
