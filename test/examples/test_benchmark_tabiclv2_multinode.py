import importlib
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"

NAN = float("nan")
INF = float("inf")


@pytest.fixture(scope="module")
def multinode() -> ModuleType:
    # The benchmark is a script, not a package module; import it the way
    # the GPU workflow's import smoke does (``PYTHONPATH=examples``).
    sys.path.insert(0, str(EXAMPLES))
    try:
        return importlib.import_module("benchmark_tabiclv2_multinode")
    finally:
        sys.path.remove(str(EXAMPLES))


@pytest.mark.parametrize(
    ("spread", "expected"),
    [
        # Rank 0 is its own reference; a finite, on-margin rank passes.
        ([[0.0, 0.0, 1.0], [0.125, 0.0025, 1.0]], True),
        # Margin ratio or top-1 out of range fails.
        ([[0.0, 0.0, 1.0], [0.5, 0.06, 1.0]], False),
        ([[0.0, 0.0, 1.0], [0.5, 0.01, 0.99]], False),
        # A NaN rank keeps top-1 above the threshold on its remaining rows
        # and Python's ``max([0.0, nan]) == 0.0`` would hide the NaN margin.
        ([[0.0, 0.0, 1.0], [NAN, NAN, 0.999]], False),
        # A single inf row leaves the median margin ratio at 0.0.
        ([[0.0, 0.0, 1.0], [INF, 0.0, 1.0]], False),
    ],
)
def test_agreement_verdict(
    multinode: ModuleType,
    spread: list[list[float]],
    expected: bool,
) -> None:
    assert multinode.agreement_verdict(torch.tensor(spread)) is expected


def test_phase_summary_reports_per_rank_throughput(
    multinode: ModuleType,
) -> None:
    # world=4 with two active ranks; inactive ranks gather zeros. Rank 1
    # served one table less over a longer window, so its throughput is
    # lower than its count alone suggests.
    counts = [98.0, 97.0, 0.0, 0.0]
    elapsed = [4.02, 4.04, 0.0, 0.0]
    phase = multinode.phase_summary(counts, elapsed, active=2)
    assert phase["tables_total"] == 195.0
    assert phase["tables_per_s"] == pytest.approx(195.0 / 4.04)
    assert phase["per_rank_tables"] == [98.0, 97.0]
    assert phase["per_rank_elapsed_s"] == [4.02, 4.04]
    assert phase["per_rank_tables_per_s"] == pytest.approx(
        [98.0 / 4.02, 97.0 / 4.04]
    )
    # The aggregate never exceeds the sum of per-rank rates.
    assert phase["tables_per_s"] <= sum(phase["per_rank_tables_per_s"])
    # Spread is the slowest rank's gap relative to the fastest one.
    fastest, slowest = 98.0 / 4.02, 97.0 / 4.04
    assert phase["per_rank_spread"] == pytest.approx(
        (fastest - slowest) / fastest
    )


def test_phase_summary_flags_lagging_rank(multinode: ModuleType) -> None:
    # One rank at 90% of the others: spread 10%, above the 5% print flag.
    counts = [90.0, 100.0, 100.0, 100.0]
    elapsed = [4.0, 4.0, 4.0, 4.0]
    phase = multinode.phase_summary(counts, elapsed, active=4)
    assert phase["per_rank_spread"] == pytest.approx(0.1)


def test_phase_summary_zero_window(multinode: ModuleType) -> None:
    phase = multinode.phase_summary([0.0], [0.0], active=1)
    assert phase["tables_per_s"] == 0.0
    assert phase["per_rank_tables_per_s"] == [0.0]
    assert phase["per_rank_spread"] == 0.0
