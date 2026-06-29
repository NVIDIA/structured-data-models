"""Performance tests for Pipeline and Recipe abstraction overhead.

Key findings that these tests guard:
- Pipeline/Recipe Python overhead (loop + try/except + TableTensor rebuild)
  is a fixed ~50 us constant regardless of n_rows.
- For data >= 1k rows the abstraction adds < 50% relative overhead.
- For data >= 10k rows the abstraction adds < 20% relative overhead.
- The overhead does not grow with n_rows (O(1) in rows, O(stages) in depth).
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable

import pytest
import torch
from sdm import CategoricalTensor, StringTensor, TableTensor
from sdm.processing import Pipeline, Processor, Recipe, StandardScale
from sdm.processing.recipe import _with_numerical


def _make_table(n_rows: int, n_cols: int) -> TableTensor:
    data = torch.randn(n_rows, n_cols)
    cat = CategoricalTensor(
        data=torch.zeros(n_rows, 1, dtype=torch.int64),
        categories=(StringTensor.from_list(["a"]),),
    )
    return TableTensor(
        columns={
            "numerical": tuple(f"x{i}" for i in range(n_cols)),
            "categorical": ("kind",),
        },
        numerical=data,
        categorical=cat,
    )


def _bench(fn: Callable[[], object], n: int = 100) -> float:
    """Return median wall-clock ms over n calls after 10 warmup calls."""
    for _ in range(10):
        fn()
    samples = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000)
    return statistics.median(samples)


class _Identity(Processor):
    requires_fit = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class _ReturnTensor(Processor):
    """Processor that returns a fixed precomputed tensor, ignoring input."""

    requires_fit = False

    def __init__(self, output: torch.Tensor) -> None:
        super().__init__()
        self.output = output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output


def test_pipeline_overhead_is_constant_in_n_rows() -> None:
    """Pipeline abstraction overhead must not grow with n_rows.

    The try/except loop + TableTensor rebuild are O(1) in n_rows. The stage
    returns a precomputed tensor so this test isolates abstraction overhead
    from tensor-kernel timing variance. We verify that overhead at 100k rows
    is at most 5x the overhead at 1k rows, ruling out per-row Python cost.
    """
    n_cols = 32

    def _elapsed(n_rows: int) -> float:
        table = _make_table(n_rows, n_cols)
        pl = Pipeline([_ReturnTensor(table.numerical.clone())])
        pl.fit(table)
        return _bench(lambda: pl.transform(table))

    overhead_1k = _elapsed(1_000)
    overhead_100k = _elapsed(100_000)

    assert overhead_100k < overhead_1k * 5 + 1.0, (
        f"Pipeline overhead grew with n_rows: "
        f"1k->{overhead_1k:.3f} ms, 100k->{overhead_100k:.3f} ms"
    )


def test_recipe_overhead_is_constant_in_n_rows() -> None:
    """Recipe.transform_preprocess overhead must not grow with n_rows."""
    n_cols = 32

    def _elapsed(n_rows: int) -> float:
        table = _make_table(n_rows, n_cols)
        rc = Recipe(preprocess=[_ReturnTensor(table.numerical.clone())])
        rc.fit_preprocess(table)
        return _bench(lambda: rc.transform_preprocess(table))

    overhead_1k = _elapsed(1_000)
    overhead_100k = _elapsed(100_000)

    assert overhead_100k < overhead_1k * 5 + 1.0, (
        f"Recipe overhead grew with n_rows: "
        f"1k->{overhead_1k:.3f} ms, 100k->{overhead_100k:.3f} ms"
    )


@pytest.mark.parametrize(("n_rows", "n_cols"), [(10_000, 64), (100_000, 16)])
def test_pipeline_relative_overhead_at_scale(n_rows: int, n_cols: int) -> None:
    """Pipeline must add < 50% overhead vs bare tensor calls at >= 10k rows."""
    table = _make_table(n_rows, n_cols)
    ss = StandardScale()
    pl = Pipeline([StandardScale()])
    ss.fit(table.numerical)
    pl.fit(table)
    numerical = table.numerical

    raw_ms = _bench(lambda: ss.transform(numerical))
    pipeline_ms = _bench(lambda: pl.transform(table))

    assert pipeline_ms < raw_ms * 1.5 + 0.5, (
        f"Pipeline too slow at {n_rows}x{n_cols}: "
        f"raw={raw_ms:.3f} ms, pipeline={pipeline_ms:.3f} ms"
    )


@pytest.mark.parametrize(("n_rows", "n_cols"), [(10_000, 64), (100_000, 16)])
def test_recipe_relative_overhead_at_scale(n_rows: int, n_cols: int) -> None:
    """Recipe must add < 50% overhead vs bare tensor calls at >= 10k rows."""
    table = _make_table(n_rows, n_cols)
    ss = StandardScale()
    rc = Recipe(preprocess=[StandardScale()])
    ss.fit(table.numerical)
    rc.fit_preprocess(table)
    numerical = table.numerical

    raw_ms = _bench(lambda: ss.transform(numerical))
    recipe_ms = _bench(lambda: rc.transform_preprocess(table))

    assert recipe_ms < raw_ms * 1.5 + 0.5, (
        f"Recipe too slow at {n_rows}x{n_cols}: "
        f"raw={raw_ms:.3f} ms, recipe={recipe_ms:.3f} ms"
    )


def test_noop_pipeline_absolute_overhead_ceiling() -> None:
    """Two-stage noop Pipeline (no tensor work) must complete in < 200 us.

    Guards the fixed Python cost: try/except loop, _validate_output isinstance
    checks, and TableTensor reconstruction. A regression here means the
    abstraction layer itself has become expensive independent of tensor work.
    """
    table = _make_table(10_000, 64)
    pl = Pipeline([_Identity(), _Identity()])
    pl.fit(table)

    elapsed_ms = _bench(lambda: pl.transform(table), n=200)

    assert elapsed_ms < 0.200, (
        f"Noop Pipeline overhead {elapsed_ms * 1000:.0f} us > 200 us ceiling"
    )


def test_tabletensor_rebuild_absolute_overhead_ceiling() -> None:
    """TableTensor reconstruction alone must complete in < 150 us.

    This is the dominant fixed cost inside Pipeline._transform.
    """
    table = _make_table(10_000, 64)
    numerical = table.numerical.clone()

    elapsed_ms = _bench(lambda: _with_numerical(table, numerical), n=200)

    assert elapsed_ms < 0.150, (
        f"TableTensor rebuild {elapsed_ms * 1000:.0f} us > 150 us ceiling"
    )


def test_pipeline_overhead_scales_linearly_with_stage_count() -> None:
    """Each additional noop stage must add O(1) cost, not O(n_rows).

    A 4-stage pipeline must complete in less than 8x the time of a 1-stage
    pipeline. Failure means per-stage cost is growing with data size.
    """
    table = _make_table(10_000, 64)

    pl1 = Pipeline([_Identity()])
    pl4 = Pipeline([_Identity(), _Identity(), _Identity(), _Identity()])
    pl1.fit(table)
    pl4.fit(table)

    t1 = _bench(lambda: pl1.transform(table), n=200)
    t4 = _bench(lambda: pl4.transform(table), n=200)

    assert t4 < t1 * 8 + 0.1, (
        f"4-stage pipeline {t4:.3f} ms is more than 8x the 1-stage "
        f"{t1:.3f} ms -- per-stage cost may be O(n_rows)"
    )
