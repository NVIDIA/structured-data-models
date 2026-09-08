from collections.abc import Callable
from typing import Any

from benchmark.tabular.system import _column_limit


def test_column_limit_uses_available_cell_budget() -> None:
    assert (
        _column_limit(
            num_rows=60_000,
            max_columns=None,
            max_cells=28_800_000,
        )
        == 480
    )
    assert (
        _column_limit(
            num_rows=30_000,
            max_columns=None,
            max_cells=28_800_000,
        )
        == 960
    )


def test_column_limit_keeps_the_stricter_rigid_limit() -> None:
    assert (
        _column_limit(
            num_rows=30_000,
            max_columns=500,
            max_cells=28_800_000,
        )
        == 500
    )
    assert (
        _column_limit(
            num_rows=60_000,
            max_columns=500,
            max_cells=28_800_000,
        )
        == 480
    )


def test_column_limit_accepts_each_independent_option() -> None:
    assert (
        _column_limit(
            num_rows=60_000,
            max_columns=256,
            max_cells=None,
        )
        == 256
    )
    assert (
        _column_limit(
            num_rows=60_000,
            max_columns=None,
            max_cells=None,
        )
        is None
    )


def test_column_limit_rejects_an_invalid_cell_budget() -> None:
    error_message = None
    try:
        _column_limit(
            num_rows=60_000,
            max_columns=None,
            max_cells=0,
        )
    except ValueError as error:
        error_message = str(error)
    assert error_message == "'max_cells' must be positive"


if __name__ == "__main__":
    tests: tuple[Callable[[], Any], ...] = (
        test_column_limit_uses_available_cell_budget,
        test_column_limit_keeps_the_stricter_rigid_limit,
        test_column_limit_accepts_each_independent_option,
        test_column_limit_rejects_an_invalid_cell_budget,
    )
    for test in tests:
        test()
