"""CPU boundary checks; the GPU model and neighbor kernel run separately."""

import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch

pytest.importorskip("relarena")
from examples.kumo.relational._relarena import adapter
from relbench.base import Database, Table, TaskType

import sdm
from sdm.processing.execution import RecipeExecution
from sdm.tensor import EnsembleTable


@pytest.mark.parametrize(
    ("binary", "all_negative"), [(False, False), (True, False), (True, True)]
)
def test_adapter_fit_predict_uses_public_sampling_and_recipe(
    monkeypatch: pytest.MonkeyPatch,
    binary: bool,
    all_negative: bool,
) -> None:
    # Keep actual seed lookup, sample materialization, and recipe execution.
    # This kernel stub returns roots only; temporal neighbor sampling is a
    # separate GPU-host integration check.
    monkeypatch.setitem(sys.modules, "pyg_lib", ModuleType("pyg_lib"))

    def sample(**kwargs: Any) -> tuple[None, None, dict[str, torch.Tensor]]:
        return (
            None,
            None,
            {
                name: torch.stack((torch.arange(seed.numel()), seed), dim=1)
                for name, seed in kwargs["seed_dict"].items()
            },
        )

    monkeypatch.setattr(
        torch.ops.pyg, "hetero_neighbor_sample", sample, raising=False
    )
    torch_proxy = SimpleNamespace(**vars(torch))
    torch_proxy.device = lambda _: torch.device("cpu")
    monkeypatch.setattr(adapter, "torch", torch_proxy)
    default_recipe = sdm.models.KumoRelational.default_recipe

    class RecipeModel:
        def __init__(self, **kwargs: Any) -> None:
            self.contexts = []
            self.queries = []

        @staticmethod
        def default_recipe() -> sdm.Recipe:
            return default_recipe()

        def fit(self, **kwargs: Any) -> None:
            assert kwargs["x"].size(0) == 8
            assert kwargs["num_estimators"] is None
            if all_negative:
                labels = kwargs["y"].flatten(0, 1).to_pandas()["label"]
                assert not labels.to_numpy().any()
            self.execution = RecipeExecution(kwargs["recipe"])
            self.contexts = self.execution.fit_transform(
                x=kwargs["x"],
                y=kwargs["y"],
                related_tables=kwargs["related_tables"],
                num_members=None,
                generator=kwargs["generator"],
            )

        def predict(
            self, x: sdm.TableTensor | EnsembleTable, related_tables: Any
        ) -> sdm.TableTensor:
            query_table = x.table(0) if isinstance(x, EnsembleTable) else x
            assert "label" not in query_table.column_names
            self.queries = self.execution.transform(x, related_tables)
            return sdm.TableTensor.from_tensor(
                torch.full((len(query_table), 1), 0.5),
                columns=(
                    "False" if all_negative else "True" if binary else "q500",
                ),
            )

    monkeypatch.setattr(sdm.models, "KumoRelational", RecipeModel)
    entities = Table(
        df=pd.DataFrame({"id": range(32), "value": np.arange(32.0)}),
        pkey_col="id",
        fkey_col_to_pkey_table={},
        time_col=None,
    )
    db = Database({"entities": entities})
    frame = pd.DataFrame(
        {
            "entity": range(32),
            "time": pd.to_datetime(["2020-01-01"] * 32),
            "label": np.arange(32) % 2 if binary else np.arange(32.0),
        }
    )
    if all_negative:
        frame["label"] = 0
    train = Table(
        df=frame,
        pkey_col=None,
        fkey_col_to_pkey_table={"entity": "entities"},
        time_col="time",
    )
    task = SimpleNamespace(
        entity_col="entity",
        entity_table="entities",
        time_col="time",
        target_col="label",
        task_type=(
            TaskType.BINARY_CLASSIFICATION if binary else TaskType.REGRESSION
        ),
    )
    model = adapter.KumoPredictor(config={})
    model.fit(task, db, train, None, seed=0)
    assert not model.sampler.data.tables["entities"].id.is_inference()
    output = model.predict(task, db, train)
    assert output.shape == (32,)
    assert output.dtype == np.float32
    if all_negative:
        np.testing.assert_array_equal(output, np.zeros(32))
    assert len(model.model.contexts) == len(model.model.queries) == 8
    for context, query in zip(
        model.model.contexts, model.model.queries, strict=True
    ):
        assert context.x.is_cpu
        assert query.x.is_cpu
        assert context.related_tables is not None
        assert query.related_tables is not None
        assert context.related_tables.tables["entities"].is_cpu
        assert query.related_tables.tables["entities"].is_cpu


@pytest.fixture
def captured_model(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Capture model inputs while retaining real CPU materialization."""
    captured: dict[str, Any] = {"queries": []}
    monkeypatch.setitem(sys.modules, "pyg_lib", ModuleType("pyg_lib"))

    def sample(**kwargs: Any) -> tuple[None, None, dict[str, torch.Tensor]]:
        return (
            None,
            None,
            {
                name: torch.stack((torch.arange(seed.numel()), seed), dim=1)
                for name, seed in kwargs["seed_dict"].items()
            },
        )

    monkeypatch.setattr(
        torch.ops.pyg, "hetero_neighbor_sample", sample, raising=False
    )
    torch_proxy = SimpleNamespace(**vars(torch))
    torch_proxy.device = lambda _: torch.device("cpu")
    monkeypatch.setattr(adapter, "torch", torch_proxy)

    class Capture:
        def __init__(self, **kwargs: Any) -> None:
            pass

        @staticmethod
        def default_recipe() -> sdm.Recipe:
            return sdm.Recipe()

        def fit(self, **kwargs: Any) -> None:
            captured.update(kwargs)

        def predict(
            self, x: EnsembleTable, related_tables: Any
        ) -> sdm.TableTensor:
            assert x.num_members == 8
            frame = x.table(0).to_pandas()
            captured["queries"].append(frame)
            return sdm.TableTensor.from_tensor(
                torch.zeros(len(frame), 1), columns=("q500",)
            )

    monkeypatch.setattr(sdm.models, "KumoRelational", Capture)
    return captured


def _fit_captured(
    frame: pd.DataFrame, config: dict[str, Any]
) -> tuple[adapter.KumoPredictor, Any, Database, Table]:
    entities = pd.DataFrame({"id": frame.entity.unique()})
    db = Database(
        {
            "entities": Table(
                df=entities,
                pkey_col="id",
                fkey_col_to_pkey_table={},
                time_col=None,
            )
        }
    )
    train = Table(
        df=frame,
        pkey_col=None,
        fkey_col_to_pkey_table={"entity": "entities"},
        time_col="time",
    )
    task = SimpleNamespace(
        entity_col="entity",
        entity_table="entities",
        time_col="time",
        target_col="label",
        task_type=TaskType.REGRESSION,
        timedelta=pd.Timedelta(days=3),
    )
    model = adapter.KumoPredictor(config=config)
    model.fit(task, db, train, None, seed=0)
    return model, task, db, train


@pytest.mark.parametrize("recent", [False, True])
def test_context_pool_selection_preserves_e8_10k(
    captured_model: dict[str, Any], recent: bool
) -> None:
    count = 100_003
    frame = pd.DataFrame(
        {
            "entity": np.arange(count)[::-1],
            "time": pd.Timestamp("2020-01-05"),
            "label": np.arange(count, dtype=np.float32),
        }
    )
    # An old timestamp at the end must not survive merely because it is last.
    frame.loc[count - 1, "time"] = pd.Timestamp("2020-01-01")
    model, _, _, _ = _fit_captured(frame, {"recent_context_pool": recent})
    expected_pool = (
        adapter._recent_context_pool(frame, "time", seed=0)
        if recent
        else frame
    )
    expected = expected_pool.entity.to_numpy()[
        adapter.context_members(expected_pool, seed=0)
    ]
    actual = captured_model["x"]["entity"].id[..., 0].numpy()
    np.testing.assert_array_equal(actual, expected)
    assert actual.shape == (8, 10_000)
    assert all(len(np.unique(member)) == 10_000 for member in actual)
    assert captured_model["num_estimators"] is None
    assert model.context_metadata["available_training_rows"] == count
    assert model.context_metadata["eligible_context_pool_rows"] == len(
        expected_pool
    )
    if recent:
        assert 0 not in actual
        assert model.context_metadata["recency_policy"] == "seeded_cutoff_ties"
    assert model.history is None


def test_recency_cutoff_ties_are_seeded_and_label_independent() -> None:
    frame = pd.DataFrame(
        {
            "entity": range(110),
            "time": pd.to_datetime(
                ["2020-01-01"] * 10 + ["2020-01-02"] * 90 + ["2020-01-03"] * 10
            ),
            "label": [1] * 30 + [0] * 80,
        }
    )
    state = torch.random.get_rng_state()
    selected = adapter._recent_context_pool(frame, "time", seed=0, limit=80)
    assert torch.equal(state, torch.random.get_rng_state())
    assert len(selected) == selected.entity.nunique() == 80
    assert set(range(100, 110)) <= set(selected.entity)
    assert selected.time.min() == pd.Timestamp("2020-01-02")
    assert selected.label.sum() > 0
    pd.testing.assert_frame_equal(
        selected, adapter._recent_context_pool(frame, "time", seed=0, limit=80)
    )
    changed_labels = frame.assign(label=frame.label * -100 + 7)
    changed = adapter._recent_context_pool(
        changed_labels, "time", seed=0, limit=80
    )
    pd.testing.assert_series_equal(selected.entity, changed.entity)
    alternate = adapter._recent_context_pool(frame, "time", seed=1, limit=80)
    assert set(selected.entity) != set(alternate.entity)


def test_recency_small_pool_preserves_input_order() -> None:
    frame = pd.DataFrame(
        {"time": pd.to_datetime(["2020-01-03", "2020-01-01"])}
    )
    pd.testing.assert_frame_equal(
        adapter._recent_context_pool(frame, "time", seed=0, limit=3), frame
    )


def test_database_insertion_order_does_not_change_relational_construction(
    captured_model: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    frames = {
        "z_entity": pd.DataFrame(
            {"id": [0, 1], "time": pd.to_datetime(["2020-01-01"] * 2)}
        ),
        "m_entity": pd.DataFrame(
            {"id": [0, 1], "time": pd.to_datetime(["2020-01-01"] * 2)}
        ),
        "a_events": pd.DataFrame(
            {"z_fk": [1, 0], "m_fk": [0, 1], "value": [4.0, 2.0]}
        ),
    }
    sampler = sdm.RelationalData.sampler
    time_columns = []

    def capture_sampler(self: sdm.RelationalData, **kwargs: Any) -> Any:
        time_columns.append(list(kwargs["time_columns"]))
        return sampler(self, **kwargs)

    monkeypatch.setattr(sdm.RelationalData, "sampler", capture_sampler)
    task = SimpleNamespace(
        entity_col="entity",
        entity_table="z_entity",
        time_col="time",
        target_col="label",
        task_type=TaskType.REGRESSION,
    )
    train = Table(
        pd.DataFrame(
            {
                "entity": [0, 1],
                "time": pd.to_datetime(["2021-01-01"] * 2),
                "label": [1.0, 2.0],
            }
        ),
        {"entity": "z_entity"},
        None,
        "time",
    )
    models = []
    for reverse in (False, True):
        names = list(frames)
        foreign_keys = [("z_fk", "z_entity"), ("m_fk", "m_entity")]
        if reverse:
            names.reverse()
            foreign_keys.reverse()
        db = Database(
            {
                name: Table(
                    frames[name],
                    dict(foreign_keys) if name == "a_events" else {},
                    None if name == "a_events" else "id",
                    None if name == "a_events" else "time",
                )
                for name in names
            }
        )
        model = adapter.KumoPredictor(config={})
        model.fit(task, db, train, None, seed=0)
        models.append(model)
    first, second = [model.sampler.data for model in models]
    assert list(first.tables) == list(second.tables) == sorted(frames)
    assert first.relationships == second.relationships
    assert time_columns == [["m_entity", "z_entity"]] * 2
    for name in frames:
        pd.testing.assert_frame_equal(
            first.tables[name].to_pandas(), second.tables[name].to_pandas()
        )
    assert first.tables["a_events"].columns[sdm.Stype.id] == ("z_fk", "m_fk")
    np.testing.assert_array_equal(
        first.tables["a_events"]["z_fk"].id[:, 0], [1, 0]
    )


def test_mature_full_training_history_reaches_context_and_query_after_recency(
    captured_model: dict[str, Any],
) -> None:
    frame = pd.DataFrame(
        {
            "entity": np.zeros(80_002, dtype=np.int64),
            "time": [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02")]
            + [pd.Timestamp("2020-01-05")] * 80_000,
            "label": [7.0, 11.0] + [99.0] * 80_000,
        }
    )
    model, task, db, train = _fit_captured(
        frame, {"recent_context_pool": True, "history_lags": 2}
    )
    assert captured_model["x"].size(0) == 8
    assert captured_model["x"].size(1) == 10_000
    assert captured_model["y"].numerical.eq(99).all()
    # Both mature labels live outside the latest-80k context pool.
    for column, value in (("history_target_1", 11), ("history_target_2", 7)):
        assert captured_model["x"][column].numerical.eq(value).all()
    query = Table(
        df=pd.DataFrame(
            {
                "entity": [0, 0, 0],
                "time": pd.to_datetime(
                    ["2020-01-06", "2020-01-04", "2020-01-05"]
                ),
                "label": [123456.0, 123456.0, 123456.0],
            }
        ),
        pkey_col=None,
        fkey_col_to_pkey_table=train.fkey_col_to_pkey_table,
        time_col="time",
    )
    model.predict(task, db, query)
    first = captured_model["queries"][-1]
    assert "label" not in first
    np.testing.assert_allclose(first.history_target_1, [11, 7, 11])
    np.testing.assert_allclose(
        first.history_target_2, [7, np.nan, 7], equal_nan=True
    )
    query.df["label"] = -999.0
    model.predict(task, db, query)
    pd.testing.assert_frame_equal(first, captured_model["queries"][-1])


@pytest.mark.parametrize("outer_refit", [False, True])
@pytest.mark.parametrize("history_lags", [0, 2])
def test_raw_event_lags_use_supplied_split_database_and_ignore_query_labels(
    captured_model: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    outer_refit: bool,
    history_lags: int,
) -> None:
    dataset = adapter.dataset_registry["rel-hm"][0]()

    def forbidden_reload(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("The adapter must not reload an uncensored DB")

    monkeypatch.setattr(dataset, "get_db", forbidden_reload)
    task = adapter.task_registry["rel-hm"]["item-sales"][0](dataset)
    events = pd.DataFrame(
        {
            "article_id": [0, 0, 0],
            "timestamp": pd.to_datetime(
                ["2020-01-07", "2020-01-14", "2020-01-20"]
            ),
            "price": [2.0, 3.0, 5.0],
        }
    ).iloc[: 3 if outer_refit else 2]
    db = Database(
        {
            "article": Table(
                pd.DataFrame({"article_id": [0]}), {}, "article_id", None
            ),
            "transactions": Table(
                events, {"article_id": "article"}, None, "timestamp"
            ),
        }
    )
    train_frame = pd.DataFrame(
        {
            "article_id": [0, 0],
            "timestamp": pd.to_datetime(["2020-01-08", "2020-01-15"]),
            "sales": [11.0, 999.0],
        }
    ).iloc[: 2 if outer_refit else 1]
    train = Table(train_frame, {"article_id": "article"}, None, "timestamp")
    model = adapter.KumoPredictor(
        config={"raw_event_lags": True, "history_lags": history_lags}
    )
    model.fit(task, db, train, None, seed=0)
    expected = np.array([2.0, 3.0])[: len(train_frame)][
        adapter.context_members(train_frame, seed=0)
    ]
    np.testing.assert_array_equal(
        captured_model["x"]["__kumo_arl0__"].numerical[..., 0], expected
    )
    assert model.context_metadata["raw_event_lag_columns"] == 10
    assert captured_model["x"].size(0) == 8
    query = Table(
        pd.DataFrame(
            {
                "article_id": [0, 0, 0],
                "timestamp": pd.to_datetime(
                    ["2020-01-23", "2020-01-09", "2020-01-16"]
                ),
                "sales": [12345.0] * 3,
            },
            index=[9, 3, 7],
        ),
        {"article_id": "article"},
        None,
        "timestamp",
    )
    model.predict(task, db, query)
    first = captured_model["queries"][-1]
    np.testing.assert_array_equal(
        first["__kumo_arl0__"], [5 if outer_refit else 0, 2, 3]
    )
    assert "sales" not in first
    assert ("history_target_1" in first) == bool(history_lags)
    query.df["sales"] = -999.0
    model.predict(task, db, query)
    pd.testing.assert_frame_equal(first, captured_model["queries"][-1])


def test_raw_event_lags_unsupported_task_is_noop(
    captured_model: dict[str, Any],
) -> None:
    frame = pd.DataFrame(
        {"entity": [0], "time": [pd.Timestamp("2020-01-01")], "label": [1.0]}
    )
    model, task, db, train = _fit_captured(frame, {"raw_event_lags": True})
    model.predict(task, db, train)
    assert model.context_metadata["raw_event_lag_columns"] == 0
    assert not any(
        name.startswith("__kumo_arl")
        for name in captured_model["x"].column_names
    )
