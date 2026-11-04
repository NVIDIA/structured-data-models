# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest


@pytest.fixture
def benchmark(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[2] / "examples")
    )
    return importlib.import_module(name="benchmark_tabiclv2")


@pytest.mark.parametrize(argnames="failure", argvalues=["timeout", "exit"])
def test_queue_continues_after_cell_failure(
    benchmark: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failure: str,
) -> None:
    out_path = tmp_path / "results.jsonl"
    submitted = []

    def run(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
        spec = json.loads(args[args.index("--run-cell") + 1])
        submitted.append(spec)
        if len(submitted) == 1:
            if failure == "timeout":
                raise subprocess.TimeoutExpired(cmd=args, timeout=1800)
            return subprocess.CompletedProcess(
                args=args, returncode=1, stderr="cell failed"
            )
        record = {**spec, "accuracy": {"pass": True}, "p50_s": 1.0}
        with out_path.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        return subprocess.CompletedProcess(args=args, returncode=0)

    monkeypatch.setattr(target=benchmark.subprocess, name="run", value=run)
    monkeypatch.setattr(
        target=benchmark.tempfile,
        name="mkdtemp",
        value=lambda **kwargs: str(tmp_path),
    )
    monkeypatch.setattr(
        target=sys,
        name="argv",
        value=["benchmark", "--out", str(out_path), "--configs", "c0-fp32"],
    )
    benchmark.main()

    records = [json.loads(line) for line in out_path.read_text().splitlines()]
    assert len(records) == len(submitted) > 1
    assert all(records[0][key] == value for key, value in submitted[0].items())
    if failure == "timeout":
        assert "timed out" in records[0]["error"]
    else:
        assert records[0]["error"] == "cell failed"
    assert all("error" not in record for record in records[1:])
    assert any(record["route"] == "fitpredict" for record in records[1:])


def test_promotion_excludes_stream_failures(
    benchmark: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    submitted = []
    timings = {
        "c0-fp32": 1.0,
        "c1-tf32": 0.001,
        "c2-bf16-autocast": 0.2,
        "c3-bf16-full": 0.3,
        "c12-autocast-compile": 0.002,
    }

    def spawn_cell(
        spec: dict[str, Any], out_path: str, workdir: str
    ) -> dict[str, Any]:
        submitted.append(spec)
        record = {
            **spec,
            "accuracy": {"pass": True},
            "p50_s": timings[spec["config"]],
        }
        if spec["config"] in {"c1-tf32", "c12-autocast-compile"}:
            record.update(
                table_stream_errors=1,
                table_stream_per_table_s=None,
                table_stream_fresh2_per_table_s=None,
            )
        else:
            record.update(
                table_stream_per_table_s=timings[spec["config"]],
                table_stream_fresh2_per_table_s=timings[spec["config"]],
            )
        return record

    monkeypatch.setattr(target=benchmark, name="spawn_cell", value=spawn_cell)
    monkeypatch.setattr(
        target=benchmark.tempfile,
        name="mkdtemp",
        value=lambda **kwargs: str(tmp_path),
    )
    monkeypatch.setattr(
        target=sys,
        name="argv",
        value=["benchmark", "--configs", ",".join(timings)],
    )
    benchmark.main()

    promoted = {
        spec["config"] for spec in submitted if spec["route"] == "fitpredict"
    }
    assert promoted == {
        "c0-fp32",
        "c2-bf16-autocast",
        "c3-bf16-full",
        "c12-autocast-compile",
    }
