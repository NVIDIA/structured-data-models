"""Optional tests for the TabArena examples."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


import pandas as pd
import pytest
import torch
sys.path.insert(0, str(Path(__file__).parents[2]))


pytest.importorskip("autogluon.core.models")

from examples.tabiclv2_tabarena.model import (  # noqa: E402
    _resolve_device,
    _prediction_to_numpy,
    _table_from_frame,
    _table_from_series,
)
from examples.tabiclv2_tabarena.run_isolated import (  # noqa: E402
    DatasetRun,
    _dataset_command,
    _dataset_slug,
    _write_campaign_report,
)
from examples.tabiclv2_tabarena.runner import (  # noqa: E402
    RunConfig,
    _prepare_output_root,
    config_from_args,
)
from sdm import Stype, infer_stypes


def test_table_conversion_preserves_feature_and_target_stypes() -> None:
    features = pd.DataFrame(
        {
            "amount": [1.0, float("nan")],
            "category": pd.Series(["a", "b"], dtype="category"),
            "enabled": [True, False],
        }
    )
    feature_table = _table_from_frame(
        features,
        stypes=infer_stypes(features),
        device=torch.device("cpu"),
    )
    classification_target = _table_from_series(
        pd.Series([10, 20], name="label"),
        name="label",
        stype=Stype.categorical,
        device=torch.device("cpu"),
    )
    regression_target = _table_from_series(
        pd.Series([1, 2], name="target"),
        name="target",
        stype=Stype.numerical,
        device=torch.device("cpu"),
    )

    assert feature_table.numerical.size(-1) == 1
    assert feature_table.categorical.size(-1) == 2
    assert classification_target.categorical.size(-1) == 1
    assert regression_target.numerical.size(-1) == 1


def test_prediction_shapes_match_autogluon_problem_types() -> None:
    probabilities = torch.tensor([[0.2, 0.8], [0.7, 0.3]]).numpy()
    quantiles = torch.tensor([[1.0, 2.0], [3.0, 5.0]]).numpy()

    assert _prediction_to_numpy(
        probabilities, problem_type="binary", class_labels=("0", "1")
    ).shape == (2,)
    assert _prediction_to_numpy(
        probabilities, problem_type="multiclass", class_labels=("0", "1")
    ).shape == (2, 2)
    assert _prediction_to_numpy(
        probabilities, problem_type="binary", class_labels=("1", "0")
    ).tolist() == pytest.approx([0.2, 0.7])
    assert _prediction_to_numpy(quantiles, problem_type="regression").tolist() == [1.5, 4.0]


def test_gpu_request_requires_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    assert _resolve_device(num_gpus=0).type == "cpu"
    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        _resolve_device(num_gpus=1)


def test_config_validation_and_output_root(tmp_path) -> None:
    config = config_from_args(
        argparse.Namespace(
            output_root=tmp_path / "run",
            num_estimators=1,
            num_cpus=1,
            num_gpus=0,
            resume=False,
            allow_partial=False,
            outer=False,
            subset=None,
            datasets=None,
        )
    )
    assert isinstance(config, RunConfig)
    assert config.output_root == (tmp_path / "run").resolve()

    _prepare_output_root(config.output_root, resume=False)
    (config.output_root / "existing").write_text("result\n")
    with pytest.raises(FileExistsError, match="--resume"):
        _prepare_output_root(config.output_root, resume=False)
    _prepare_output_root(config.output_root, resume=True)




def test_isolated_campaign_command_and_manual_report(tmp_path) -> None:
    config = RunConfig(
        output_root=tmp_path,
        num_estimators=1,
        num_cpus=1,
        num_gpus=1,
        resume=False,
        allow_partial=False,
        outer=True,
        subset=None,
        datasets=None,
    )
    completed_root = tmp_path / "datasets" / "complete"
    dataset_report_dir = completed_root / "report"
    dataset_report_dir.mkdir(parents=True)
    pd.DataFrame(
        [{"dataset": "complete", "problem_type": "binary", "metric_error": 0.5}]
    ).to_csv(dataset_report_dir / "results_per_split.csv", index=False)

    command = _dataset_command(config, "complete", completed_root)
    assert command[command.index("--num-estimators") + 1] == "1"
    assert command[command.index("--datasets") + 1] == "complete"
    assert "--outer" in command

    _write_campaign_report(
        tmp_path,
        [
            DatasetRun("complete", completed_root, 0, True),
            DatasetRun("failed dataset", tmp_path / "datasets" / "failed", 1, False),
        ],
        excluded_datasets={"APSFailure"},
    )
    assert (tmp_path / "report" / "results_per_split.csv").is_file()
    status = json.loads((tmp_path / "report" / "campaign_status.json").read_text())
    assert status["completed_datasets"] == ["complete"]
    assert status["failed_datasets"][0]["dataset"] == "failed dataset"
    assert _dataset_slug("failed dataset/1") == "failed-dataset-1"
@pytest.mark.skipif(
    os.environ.get("SDM_RUN_TABARENA_SMOKE") != "1",
    reason="Set SDM_RUN_TABARENA_SMOKE=1 to run the real TabArena smoke test",
)
def test_real_tabarena_smoke(tmp_path) -> None:
    """Run one binary, multiclass, and regression TabArena task end to end."""
    from examples.tabiclv2_tabarena.runner import run

    run(
        RunConfig(
            output_root=tmp_path / "tabarena-smoke",
            num_estimators=1,
            num_cpus=1,
            num_gpus=1 if torch.cuda.is_available() else 0,
            resume=False,
            outer=True,
            allow_partial=False,
            subset=["lite"],
            datasets=[
                "blood-transfusion-service-center",
                "anneal",
                "QSAR_fish_toxicity",
            ],
        ),
        debug_mode=True,
    )

    report_dir = tmp_path / "tabarena-smoke" / "report"
    assert (report_dir / "results_per_split.csv").is_file()
    assert (report_dir / "metric_summary.csv").is_file()
    assert (report_dir / "report_status.json").is_file()
