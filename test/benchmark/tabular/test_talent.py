# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import runpy
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest


def test_result_identities(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    talent = ModuleType("TALENT")
    models = ModuleType("benchmark.tabular.talent.models")

    def run(*_: Any, **__: Any) -> SimpleNamespace:
        return SimpleNamespace(to_dict=lambda: {"score": 0.8})

    monkeypatch.setattr(talent, "run_from_dataset", run, raising=False)
    monkeypatch.setattr(
        models,
        "MODEL_CONFIGS",
        {
            key: SimpleNamespace(
                name=name, num_estimators=8, low_cardinality="infer"
            )
            for key, name in [
                ("tabiclv2", "TabICLv2"),
                ("kumo-tabular-small", "KumoTabular-Small"),
                ("kumo-tabular-small-ft", "KumoTabularSmallFT"),
            ]
        },
        raising=False,
    )
    monkeypatch.setattr(
        models, "register_sdm_method", lambda: None, raising=False
    )
    monkeypatch.setattr(
        models, "UnsupportedDatasetError", RuntimeError, raising=False
    )
    monkeypatch.setitem(sys.modules, "TALENT", talent)
    monkeypatch.setitem(sys.modules, models.__name__, models)

    for model in models.MODEL_CONFIGS:
        for finetune in [False, True]:
            argv = [
                "talent",
                "--model",
                model,
                "--dataset-path",
                str(tmp_path),
                "--dataset",
                "example",
                "--output-dir",
                str(tmp_path),
            ]
            if finetune:
                argv.append("--finetune")
            monkeypatch.setattr(sys, "argv", argv)
            runpy.run_module(
                "benchmark.tabular.talent.main", run_name="__main__"
            )

    records = [
        json.loads(path.read_text())
        for path in tmp_path.glob("*/example/result.json")
    ]
    assert len(records) == 6
    assert len({record["model"] for record in records}) == 6
    assert len({record["method"] for record in records}) == 6
    for record in records:
        general = record["config"]["general"]
        assert record["model"] == general["model"] + (
            "-finetuned" if general["finetune"] else ""
        )
        assert record["method"].endswith("-FT") == general["finetune"]
