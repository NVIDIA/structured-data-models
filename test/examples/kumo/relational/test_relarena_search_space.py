import json
from itertools import combinations, product

import pytest

pytest.importorskip("relarena")

from examples.kumo.relational._relarena.adapter import (
    CONTEXT_ROWS,
    DEFAULT,
    ESTIMATORS,
)
from examples.kumo.relational._relarena.search_space import SEARCH_SPACE
from relarena.model import RelArenaModel
from relarena.registry import registry
from relarena.search_space import SearchSpace
from relarena.system import RelArenaSystem

from examples.kumo.relational import rel_arena, rel_arena_system


def test_shared_grid_is_default_first_unique_and_seed_independent():
    assert isinstance(SEARCH_SPACE, SearchSpace)
    configs = SEARCH_SPACE.configs(n_trials=30, seed=0)
    assert len(configs) == 30
    assert configs[0] == SEARCH_SPACE.default_overrides == DEFAULT
    assert (
        len({json.dumps(config, sort_keys=True) for config in configs}) == 30
    )
    assert SEARCH_SPACE.configs(n_trials=30, seed=97) == configs
    assert SEARCH_SPACE.configs(n_trials=3, seed=97) == configs[:3]


@pytest.mark.parametrize("order", [2, 3])
def test_grid_covers_every_pair_and_triple_of_six_factors(order):
    # Raw-event and training-label history form one three-level factor.
    rows = [
        (
            tuple(config["num_neighbors"]),
            (config["raw_event_lags"], config["history_lags"]),
            config["text_pca_components"],
            config["regression_transform"],
            config["recent_context_pool"],
            config["entity_timestamps"],
        )
        for config in SEARCH_SPACE.configs(n_trials=30, seed=0)
    ]
    levels = [
        {(1,), (1, 1), (16,), (16, 16), (64, 64)},
        {(False, 0), (True, 0), (False, 2)},
        {0, 32},
        {"standard", "quantile"},
        {False, True},
        {"retain", "drop"},
    ]
    for index, expected in enumerate(levels):
        assert {row[index] for row in rows} == expected
    for indices in combinations(range(6), order):
        observed = {tuple(row[index] for index in indices) for row in rows}
        assert observed == set(product(*(levels[index] for index in indices)))


def test_grid_preserves_fixed_engine_settings():
    assert CONTEXT_ROWS == 10_000
    assert ESTIMATORS == 8
    fixed = {
        "cache_text": True,
        "text_cache_max_bytes": 32 * 1024**3,
        "max_keys": 20_000,
        "temporal_strategy": "last",
    }
    for config in SEARCH_SPACE.configs(n_trials=30, seed=0):
        assert set(config) == set(DEFAULT)
        assert {key: config[key] for key in fixed} == fixed


def test_native_registration_uses_one_task_independent_model_space():
    model = rel_arena.KumoModel
    system = rel_arena_system.KumoSystem
    assert issubclass(model, RelArenaModel)
    assert issubclass(system, RelArenaSystem)
    assert registry.get(model.name) is model
    assert registry.get(system.name) is system
    assert registry.kind(model.name) == "model"
    assert registry.kind(system.name) == "system"
    # A concrete shared space, not a task-aware factory or per-task lookup.
    assert registry.search_space(model.name) is SEARCH_SPACE
    assert rel_arena_system.SEARCH_SPACE is SEARCH_SPACE
    with pytest.raises(TypeError, match="no harness search space"):
        registry.search_space(system.name)
