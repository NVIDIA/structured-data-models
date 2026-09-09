from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("relarena")

from relarena.dataset import InnerSplit, OuterSplit
from relarena.metrics import mae
from relarena.search_space import SearchSpace
from relbench.base import Database, Table, TaskType

from examples.kumo.relational import rel_arena_system as example


@pytest.fixture
def procedure(monkeypatch):
    def table(ids, labels=None):
        frame = pd.DataFrame(
            {"entity": ids, "time": pd.Timestamp("2020-01-01")}
        )
        if labels is not None:
            frame["target"] = labels
        return Table(
            frame,
            fkey_col_to_pkey_table={"entity": "entities"},
            pkey_col=None,
            time_col="time",
        )

    train = table([1, 2], [2.0, 4.0])
    val = table([3, 4], [1.0, 1.0])
    query = table([9, 7])
    inner = InnerSplit(
        Database({}), pd.Timestamp("2020-01-01"), train, val, val
    )
    outer = OuterSplit(
        Database({}), pd.Timestamp("2021-01-01"), train, query, val
    )
    fitted = []
    clock = [0.0]
    state = {"fit_seconds": 0.0, "predict_seconds": 0.0, "fail": False}

    class Predictor:
        name = "fixture-predictor"

        def __init__(self, config, **kwargs):
            self.config = config

        def fit(self, task, db, train_table, val_table, **kwargs):
            clock[0] += state["fit_seconds"]
            if state["fail"]:
                raise RuntimeError("fixture candidate failed")
            fitted.append((db, train_table.df.copy(), val_table, kwargs))

        def predict(self, task, db, table):
            clock[0] += state["predict_seconds"]
            if db is inner.db_state:
                return np.full(len(table), self.config["value"])
            assert "target" not in table.df
            # The VAL winner does not minimize these outer values.
            return table.df.entity.to_numpy() + 100 * self.config["value"]

    def evaluate(predictions, target_table, metrics):
        assert target_table is val  # No hidden outer labels may be requested.
        assert len(predictions) == len(val)
        return {"mae": float(np.abs(predictions - val.df.target).mean())}

    task = SimpleNamespace(
        task_type=TaskType.REGRESSION, metrics=[mae], evaluate=evaluate
    )
    monkeypatch.setattr(example, "KumoPredictor", Predictor)
    monkeypatch.setattr(
        example,
        "SEARCH_SPACE",
        SearchSpace(
            default_overrides={"value": 0},
            fixed_grid=[{"value": 0}, {"value": 1}],
        ),
    )
    monkeypatch.setattr(example.time, "monotonic", lambda: clock[0])
    return task, inner, outer, fitted, state


def test_system_selects_full_validation_and_refits_train_val(procedure):
    task, inner, outer, fitted, _ = procedure
    system = example.KumoSystem()
    predictions = system.run(
        task, inner_split=inner, outer_split=outer, seed=0
    )

    assert system.selected_config == {"value": 1}
    assert system.validation_scores == [1.0, 0.0]
    np.testing.assert_array_equal(predictions, [109, 107])
    for db, frame, _, _ in fitted[:-1]:
        assert db is inner.db_state
        pd.testing.assert_frame_equal(frame, inner.train_table.df)
    db, frame, val, _ = fitted[-1]
    assert db is outer.db_state
    pd.testing.assert_frame_equal(
        frame,
        pd.concat(
            [outer.train_table.df, outer.val_table.df], ignore_index=True
        ),
    )
    assert val is None
    assert (
        len(outer.train_table) == 2
    )  # Refit does not mutate the input split.


def test_system_budget_covers_all_trials_and_final_prediction(procedure):
    task, inner, outer, _, state = procedure
    state["predict_seconds"] = 4.0
    with pytest.raises(TimeoutError, match="budget exhausted"):
        example.KumoSystem().run(
            task, inner_split=inner, outer_split=outer, seed=0, time_limit=10.0
        )


def test_system_passes_remaining_budget_to_each_fit(procedure):
    task, inner, outer, fitted, state = procedure
    state["fit_seconds"] = 2.0
    example.KumoSystem().run(
        task, inner_split=inner, outer_split=outer, seed=0, time_limit=10.0
    )
    assert [entry[3]["time_limit"] for entry in fitted] == [10.0, 8.0, 6.0]


def test_system_rejects_failed_candidate_instead_of_partial_selection(
    procedure,
):
    task, inner, outer, _, state = procedure
    state["fail"] = True
    with pytest.raises(RuntimeError, match="did not complete validation"):
        example.KumoSystem().run(
            task, inner_split=inner, outer_split=outer, seed=0
        )
