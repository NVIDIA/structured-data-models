# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib
import json
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


def test_agreement_input_failure(
    multinode: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*args: object, **kwargs: object) -> None:
        raise RuntimeError("Input allocation failed")

    monkeypatch.setattr(multinode, "make_table", fail)
    assert multinode.rank_agreement_gate(
        model=None, rank=0, world=1, device=torch.device("cpu")
    ) == (False, [])


@pytest.mark.parametrize(
    argnames="operation",
    argvalues=["clone", "topk", "zeros_like", "tolist"],
)
def test_agreement_local_failure(
    multinode: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    x, y = torch.ones(5, 2), torch.tensor([0, 1, 0])
    logits = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    monkeypatch.setattr(
        target=multinode,
        name="make_table",
        value=lambda *args, **kwargs: (x, y),
    )
    monkeypatch.setattr(
        target=multinode,
        name="pad_to_buckets",
        value=lambda x, y: (
            x,
            y,
            {"seqused_train": torch.tensor(3, dtype=torch.int32)},
            2,
        ),
    )
    monkeypatch.setattr(
        target=multinode, name="call", value=lambda *args, **kwargs: logits
    )

    def fail(*args: object, **kwargs: object) -> None:
        raise torch.OutOfMemoryError(f"Injected {operation} failure")

    monkeypatch.setattr(
        target=torch if operation == "zeros_like" else torch.Tensor,
        name=operation,
        value=fail,
    )
    assert multinode.rank_agreement_gate(
        model=None, rank=0, world=1, device=torch.device("cpu")
    ) == (False, [])


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


@pytest.mark.parametrize(
    argnames="operation", argvalues=["tensor", "zeros_like", "item"]
)
def test_gather_stats_local_failure(
    multinode: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    ready = torch.ones(())
    target = torch.Tensor if operation == "item" else torch
    original = getattr(target, operation)
    destroyed = []

    def fail(*args: object, **kwargs: object) -> object:
        tensor = args[0] if args else None
        if operation != "item" or (
            isinstance(tensor, torch.Tensor) and tensor.shape == (1,)
        ):
            raise torch.OutOfMemoryError(f"Injected {operation} failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(target=target, name=operation, value=fail)
    monkeypatch.setattr(
        target=multinode.dist, name="get_rank", value=lambda: 0
    )
    monkeypatch.setattr(
        target=multinode.dist, name="all_reduce", value=lambda *a, **kw: None
    )
    monkeypatch.setattr(
        target=multinode.dist, name="all_gather", value=lambda *a, **kw: None
    )
    monkeypatch.setattr(
        target=multinode.dist,
        name="destroy_process_group",
        value=lambda: destroyed.append(True),
    )
    with pytest.raises(SystemExit, match="1"):
        multinode.gather_stats(
            value=1.0, world=2, device=torch.device("cpu"), ready=ready
        )
    assert destroyed


@pytest.mark.parametrize(
    argnames="operation", argvalues=["all_reduce", "all_gather"]
)
def test_gather_stats_transport_failure(
    multinode: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    trace = []

    def reduce(*args: object, **kwargs: object) -> None:
        trace.append("all_reduce")
        if operation == "all_reduce":
            raise RuntimeError("Transport failed")

    def gather(*args: object, **kwargs: object) -> None:
        trace.append("all_gather")
        raise RuntimeError("Transport failed")

    monkeypatch.setattr(
        target=multinode.dist, name="get_rank", value=lambda: 0
    )
    monkeypatch.setattr(target=multinode.dist, name="all_reduce", value=reduce)
    monkeypatch.setattr(target=multinode.dist, name="all_gather", value=gather)
    with pytest.raises(RuntimeError, match="Transport failed"):
        multinode.gather_stats(
            value=1.0,
            world=2,
            device=torch.device("cpu"),
            ready=torch.ones(()),
        )
    assert trace == (
        ["all_reduce"]
        if operation == "all_reduce"
        else ["all_reduce", "all_gather"]
    )


@pytest.mark.parametrize(
    argnames=("value", "varlen_graphs"),
    argvalues=[
        (NAN, None),
        (INF, None),
        (-INF, None),
        (1.0, None),
        (1.0, 1),
        (1.0, 0),
    ],
)
def test_main_writes_strict_json(
    multinode: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    value: float,
    varlen_graphs: int | None,
) -> None:
    x, y = torch.ones(5, 2), torch.tensor([0, 1, 0])
    logits = torch.tensor([[value, 0.0], [0.0, 1.0]])
    timings = {"load_s": 1.0, "compile_s": 2.0}
    if varlen_graphs is not None:
        timings.update(
            varlen_active=1.0,
            varlen_graphs_built=float(varlen_graphs),
            varlen_build_failures=0.0,
        )
        frontend = ModuleType("cudnn")
        frontend.__dict__["__version__"] = "test"
        monkeypatch.setitem(sys.modules, "cudnn", frontend)
    monkeypatch.setattr(
        target=multinode,
        name="init_distributed",
        value=lambda duration: (0, 1, torch.device("cpu")),
    )
    monkeypatch.setattr(
        target=multinode,
        name="prepare_model",
        value=lambda *a, **kw: (object(), timings),
    )
    monkeypatch.setattr(
        target=multinode,
        name="make_table",
        value=lambda *a, **kw: (x, y),
    )
    monkeypatch.setattr(
        target=multinode,
        name="pad_to_buckets",
        value=lambda x, y: (
            x,
            y,
            {"seqused_train": torch.tensor(3, dtype=torch.int32)},
            2,
        ),
    )
    monkeypatch.setattr(
        target=multinode, name="call", value=lambda *a, **kw: logits
    )
    monkeypatch.setattr(
        target=multinode, name="serve_tables", value=lambda **kw: (2, 1.0)
    )
    monkeypatch.setattr(
        target=torch.cuda, name="get_device_name", value=lambda device: "test"
    )
    monkeypatch.setattr(
        target=torch, name="set_float32_matmul_precision", value=lambda p: None
    )
    monkeypatch.setattr(
        target=multinode.atexit, name="register", value=lambda *a, **kw: None
    )
    monkeypatch.setattr(
        target=multinode.signal, name="signal", value=lambda *a, **kw: None
    )
    monkeypatch.setattr(
        target=multinode.tempfile,
        name="mkdtemp",
        value=lambda **kw: str(tmp_path),
    )
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(tmp_path))
    path = tmp_path / "report.json"
    argv = ["benchmark", "--out", str(path)]
    if varlen_graphs is not None:
        argv.append("--use-varlen")
    monkeypatch.setattr(target=sys, name="argv", value=argv)
    if value == 1.0 and varlen_graphs != 0:
        multinode.main()
    else:
        with pytest.raises(SystemExit, match="1"):
            multinode.main()
    if varlen_graphs == 0:
        assert not path.exists()
        return
    report = json.loads(path.read_text())
    json.dumps(report, allow_nan=False)
    assert report["rank_agreement"] == (value == 1.0)
    assert report["startup"]["load_s"] == [1.0]
    assert report["use_varlen"] == (varlen_graphs is not None)
    if value != 1.0:
        assert None in report["rank_agreement_stats"][0]
        assert "phases" not in report
