"""Optional tests for the local TabArena example."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[2]))


from examples.tabiclv2_tabarena.run_local import (
    RunConfig,
    _prepare_output_root,
    _run_jobs,
    _write_results_report,
    config_from_args,
)
from sdm import Stype, infer_stypes


def test_table_conversion_preserves_feature_and_target_stypes() -> None:
    pytest.importorskip("autogluon.core.models")
    from examples.tabiclv2_tabarena.model import (
        _table_from_frame,
        _table_from_series,
    )

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
    pytest.importorskip("autogluon.core.models")
    from examples.tabiclv2_tabarena.model import _prediction_to_numpy

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
    assert _prediction_to_numpy(
        quantiles, problem_type="regression"
    ).tolist() == [1.5, 4.0]


def test_gpu_request_requires_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("autogluon.core.models")
    from examples.tabiclv2_tabarena.model import _resolve_device

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    assert _resolve_device(num_gpus=0).type == "cpu"
    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        _resolve_device(num_gpus=1)


def test_config_validation_and_output_root(tmp_path: Path) -> None:
    config = config_from_args(
        argparse.Namespace(
            output_root=tmp_path / "run",
            num_estimators=1,
            precision="auto",
            num_cpus=1,
            num_gpus=0,
            outer=False,
            subset=None,
            datasets=None,
        )
    )
    assert isinstance(config, RunConfig)
    assert config.output_root == (tmp_path / "run").resolve()
    assert config.precision == "auto"

    _prepare_output_root(config.output_root)
    (config.output_root / "existing").write_text("result\n")
    with pytest.raises(FileExistsError, match="fresh path"):
        _prepare_output_root(config.output_root)


def test_local_dispatch_and_results_report(tmp_path: Path) -> None:
    class Context:
        def run_jobs(self, jobs, **kwargs):
            self.jobs = jobs
            self.kwargs = kwargs
            return [{"completed": True}]

    context = Context()
    jobs = [{"task": "example"}]
    assert _run_jobs(context, jobs, output_root=tmp_path) == [
        {"completed": True}
    ]
    assert context.jobs == jobs
    assert context.kwargs == {
        "expname": tmp_path,
        "new_result_prefix": "[SDM] ",
        "debug_mode": True,
    }

    results = pd.DataFrame([{"dataset": "example", "metric_error": 0.5}])
    _write_results_report(
        results,
        tmp_path / "report",
        precision="bf16",
    )
    report = pd.read_csv(tmp_path / "report" / "results_per_split.csv")
    assert report.precision.tolist() == ["bf16"]


def test_precision_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("autogluon.core.models")
    from examples.tabiclv2_tabarena.model import _autocast_enabled

    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)

    assert not _autocast_enabled(
        precision="auto",
        device=torch.device("cpu"),
    )
    assert not _autocast_enabled(
        precision="fp32",
        device=torch.device("cuda"),
    )
    assert _autocast_enabled(
        precision="auto",
        device=torch.device("cuda"),
    )
    assert _autocast_enabled(
        precision="bf16",
        device=torch.device("cuda"),
    )
    with pytest.raises(ValueError, match="requires a CUDA device"):
        _autocast_enabled(
            precision="bf16",
            device=torch.device("cpu"),
        )
    with pytest.raises(ValueError, match="must be one of"):
        _autocast_enabled(
            precision="float16",
            device=torch.device("cuda"),
        )


@pytest.mark.skipif(
    os.environ.get("SDM_RUN_TABARENA_SMOKE") != "1",
    reason="Set SDM_RUN_TABARENA_SMOKE=1 to run the real TabArena smoke test",
)
def test_real_tabarena_smoke(tmp_path: Path) -> None:
    """Run one binary, multiclass, and regression task end to end."""
    pytest.importorskip("autogluon.core.models")
    pytest.importorskip("tabarena")
    from examples.tabiclv2_tabarena.run_local import run

    run(
        RunConfig(
            output_root=tmp_path / "tabarena-smoke",
            num_estimators=1,
            precision="auto",
            num_cpus=1,
            num_gpus=1 if torch.cuda.is_available() else 0,
            outer=True,
            subset=["lite"],
            datasets=[
                "blood-transfusion-service-center",
                "anneal",
                "QSAR_fish_toxicity",
            ],
        )
    )

    report = tmp_path / "tabarena-smoke" / "report" / "results_per_split.csv"
    assert report.is_file()
