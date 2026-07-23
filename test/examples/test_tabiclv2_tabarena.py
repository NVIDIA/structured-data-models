"""Optional tests for the local TabArena example."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Literal

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
            num_cpus=1,
            num_gpus=0,
            mode="autogluon-compatible",
            outer=False,
            subset=None,
            datasets=None,
        )
    )
    assert isinstance(config, RunConfig)
    assert config.output_root == (tmp_path / "run").resolve()

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
        mode="sdm-native",
    )
    report = pd.read_csv(tmp_path / "report" / "results_per_split.csv")
    assert report.integration_mode.tolist() == ["sdm-native"]


def test_sdm_native_schema_and_class_label_contract() -> None:
    pytest.importorskip("tabarena")
    from examples.tabiclv2_tabarena.sdm_system import (
        SDMTabICLv2System,
        _align_features,
        _class_labels_by_key,
        _fit_feature_schema,
        _labels_from_prediction_columns,
        _order_probabilities_for_tabarena,
        _tabarena_class_order,
    )

    frame = pd.DataFrame(
        {
            "amount": [1.0, 2.0],
            "segment": pd.Series(["a", "b"], dtype="category"),
            "user_id": [10, 20],
        }
    )
    schema = _fit_feature_schema(frame)
    aligned = _align_features(
        frame.loc[:, ["user_id", "segment", "amount"]],
        schema=schema,
    )

    assert aligned.columns.tolist() == ["amount", "segment"]
    assert schema.id_columns == ("user_id",)
    assert _align_features(
        frame.drop(columns="user_id"),
        schema=schema,
    ).columns.tolist() == ["amount", "segment"]
    assert not SDMTabICLv2System.preprocess_data
    assert not SDMTabICLv2System.preprocess_label

    with pytest.raises(ValueError, match="unexpected columns"):
        _align_features(frame.assign(extra=1), schema=schema)
    with pytest.raises(ValueError, match="semantic types changed"):
        _align_features(frame.assign(amount=["one", "two"]), schema=schema)
    with pytest.raises(ValueError, match="datetime columns"):
        _fit_feature_schema(
            pd.DataFrame({"timestamp": pd.to_datetime(["2024-01-01"])})
        )

    string_labels = _class_labels_by_key(pd.Series(["U", "R"]))
    assert _labels_from_prediction_columns(
        ("U", "R"), labels_by_key=string_labels
    ) == ["U", "R"]
    integer_labels = _class_labels_by_key(pd.Series([10, 20]))
    assert _labels_from_prediction_columns(
        ("20", "10"), labels_by_key=integer_labels
    ) == [20, 10]
    with pytest.raises(ValueError, match="ambiguous string representations"):
        _class_labels_by_key(pd.Series([1, "1"]))

    tabarena_labels = pd.Series(
        pd.Categorical(
            ["3", "2", "5", "U", "1"],
            categories=["1", "2", "3", "5", "U"],
        )
    )
    class_order = _tabarena_class_order(
        tabarena_labels,
        problem_type="multiclass",
    )
    assert class_order == ("1", "2", "3", "5", "U")
    probabilities = pd.DataFrame(
        [[0.30, 0.04, 0.05, 0.20, 0.41]],
        columns=pd.Index(["3", "U", "5", "2", "1"]),
    )
    ordered = _order_probabilities_for_tabarena(
        probabilities,
        class_order=class_order,
    )
    assert ordered.columns.tolist() == ["1", "2", "3", "5", "U"]
    assert ordered.iloc[0].tolist() == pytest.approx(
        [0.41, 0.20, 0.30, 0.05, 0.04]
    )

    with pytest.raises(RuntimeError, match="missing labels"):
        _order_probabilities_for_tabarena(
            probabilities.drop(columns="U"),
            class_order=class_order,
        )

    boolean_probabilities = pd.DataFrame(
        [[0.8, 0.2]],
        columns=pd.Index([True, False], dtype=object),
    )
    ordered_boolean = _order_probabilities_for_tabarena(
        boolean_probabilities,
        class_order=(False, True),
    )
    assert ordered_boolean.columns.tolist() == [False, True]
    assert ordered_boolean.iloc[0].tolist() == pytest.approx([0.2, 0.8])


@pytest.mark.skipif(
    os.environ.get("SDM_RUN_TABARENA_SMOKE") != "1",
    reason="Set SDM_RUN_TABARENA_SMOKE=1 to run the real TabArena smoke test",
)
@pytest.mark.parametrize("mode", ["autogluon-compatible", "sdm-native"])
def test_real_tabarena_smoke(
    tmp_path: Path,
    mode: Literal["autogluon-compatible", "sdm-native"],
) -> None:
    """Run one binary, multiclass, and regression task end to end."""
    pytest.importorskip("autogluon.core.models")
    pytest.importorskip("tabarena")
    from examples.tabiclv2_tabarena.run_local import run

    run(
        RunConfig(
            output_root=tmp_path / f"tabarena-smoke-{mode}",
            num_estimators=1,
            num_cpus=1,
            num_gpus=1 if torch.cuda.is_available() else 0,
            mode=mode,
            outer=True,
            subset=["lite"],
            datasets=[
                "blood-transfusion-service-center",
                "anneal",
                "QSAR_fish_toxicity",
            ],
        )
    )

    report = (
        tmp_path
        / f"tabarena-smoke-{mode}"
        / "report"
        / "results_per_split.csv"
    )
    assert report.is_file()
