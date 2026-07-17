from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

pytest.importorskip("autogluon")
pytest.importorskip("tabarena")

from examples.benchmarking import run_tabiclv2_local_comparison as runner
from sdm.processing import (
    CategoricalAlign,
    ConstantFilter,
    Identity,
    MeanImpute,
    Sequential,
    SigmaClip,
    StandardScale,
)

pytestmark = pytest.mark.tabarena


def _fake_worker(spec: Mapping[str, Any]) -> dict[str, Any]:
    implementation = spec["implementation"]
    profile = spec["profile"]
    base = 2.0 if implementation == runner.ORIGINAL else 1.0
    if profile == runner.LOCAL_NATIVE_DEFAULT:
        base += 2.0
    inference_repeats = int(spec["inference_repeats"])
    return {
        "profile": profile,
        "implementation": implementation,
        "trial": spec["trial"],
        "execution_order": spec["execution_order"],
        "seed": spec["seed"],
        "dataset": spec["dataset"],
        "task_id": spec["task_id"],
        "tabarena_split": 0,
        "repeat": spec["repeat"],
        "fold": spec["fold"],
        "sample": spec["sample"],
        "train_rows": 604,
        "test_rows": 303,
        "features": 6,
        "metric": "rmse",
        "metric_error": 0.9 + base / 100,
        "fit_time_s": base,
        "first_inference_time_s": base / 10,
        "inference_times_s": [
            base / 100 + repeat / 1000 for repeat in range(inference_repeats)
        ],
    }


def _base_spec(*, inference_repeats: int = 3) -> dict[str, Any]:
    return {
        "checkpoint_path": "/tmp/checkpoint.ckpt",
        "checkpoint_sha256": "0" * 64,
        "seed": 0,
        "num_cpus": 1,
        "num_gpus": 0,
        "task_id": runner.DEFAULT_TASK_ID,
        "dataset": runner.DEFAULT_DATASET,
        "repeat": 0,
        "fold": 0,
        "sample": 0,
        "inference_repeats": inference_repeats,
    }


def _historical_frame(*, duplicate: bool = False) -> pd.DataFrame:
    row = {
        "dataset": runner.DEFAULT_DATASET,
        "fold": 0,
        "method": "TABICLV2 (default)",
        "metric_error": 0.8942657,
        "time_train_s": 6.86488,
        "time_infer_s": 0.439847,
        "metric": "rmse",
        "problem_type": "regression",
        "method_subtype": runner.HISTORICAL_METHOD_SUBTYPE,
        "config_type": runner.HISTORICAL_CONFIG_TYPE,
        "ta_suite": "tabarena-test-suite",
    }
    rows = [row, dict(row)] if duplicate else [row]
    return pd.DataFrame(rows)


def _historical_summary() -> dict[str, object]:
    return runner.select_historical_baseline(
        _historical_frame(),
        dataset=runner.DEFAULT_DATASET,
        tabarena_split=0,
        task_id=runner.DEFAULT_TASK_ID,
        repeat=0,
        fold=0,
        sample=0,
        suite="tabarena-test-suite",
        config="TabICLv2_c1_BAG_L1",
    )


def _local_results(
    *,
    trials: int = 2,
    inference_repeats: int = 3,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    return runner.run_local_benchmarks(
        _base_spec(inference_repeats=inference_repeats),
        warmup_pairs=0,
        trials=trials,
        worker_runner=_fake_worker,
    )


def test_matched_and_native_configurations_are_explicit() -> None:
    checkpoint = Path("/tmp/regressor.ckpt")
    matched_original = runner.resolved_configuration(
        profile=runner.MATCHED_PARITY,
        implementation=runner.ORIGINAL,
        checkpoint_path=checkpoint,
        seed=7,
    )
    assert matched_original == {
        "model_path": str(checkpoint),
        "allow_auto_download": False,
        "n_estimators": 1,
        "norm_methods": "none",
        "feat_shuffle_method": "none",
        "outlier_threshold": 4.0,
        "batch_size": 1,
        "kv_cache": "kv",
        "random_state": 7,
        "use_amp": False,
        "use_fa3": False,
        "offload_mode": False,
    }

    native_original = runner.resolved_configuration(
        profile=runner.LOCAL_NATIVE_DEFAULT,
        implementation=runner.ORIGINAL,
        checkpoint_path=checkpoint,
        seed=7,
    )
    assert native_original["n_estimators"] == 8
    assert native_original["norm_methods"] == ["none", "power"]
    assert native_original["feat_shuffle_method"] == "latin"
    assert native_original["batch_size"] == 8
    assert native_original["kv_cache"] is False
    assert native_original["random_state"] == 42
    assert native_original["use_amp"] == "auto"
    assert native_original["use_fa3"] == "auto"
    assert native_original["offload_mode"] == "auto"

    matched_sdm = runner.resolved_configuration(
        profile=runner.MATCHED_PARITY,
        implementation=runner.SDM,
        checkpoint_path=checkpoint,
        seed=7,
    )
    assert matched_sdm["checkpoint_path"] == str(checkpoint)
    assert "model_path" not in matched_sdm
    assert matched_sdm["num_estimators"] == 1
    assert matched_sdm["seed"] == 7
    assert matched_sdm["cache"] is True
    recipe = matched_sdm["recipe"]
    assert isinstance(recipe, list)
    assert "ordinal_encode_sorted_categories_then_numerical" in recipe
    assert "fixed_clip_-100_100" in recipe

    native_sdm = runner.resolved_configuration(
        profile=runner.LOCAL_NATIVE_DEFAULT,
        implementation=runner.SDM,
        checkpoint_path=checkpoint,
        seed=7,
    )
    assert native_sdm["recipe"] == "sdm_tabiclv2_default_recipe"
    assert native_sdm["num_estimators"] == 1


def test_matched_recipe_has_reference_single_estimator_steps() -> None:
    recipe = runner.matched_parity_recipe()
    assert isinstance(recipe.features, Sequential)
    assert [type(step) for step in recipe.features.steps] == [
        runner._ReferenceCategoricalEncoding,
        MeanImpute,
        ConstantFilter,
        StandardScale,
        runner._FixedRangeClip,
        Identity,
        SigmaClip,
        Identity,
    ]
    categorical = recipe.features.steps[0]
    assert isinstance(categorical, runner._ReferenceCategoricalEncoding)
    assert isinstance(categorical.categorical_align, CategoricalAlign)
    assert categorical.categorical_align.category_order == "sorted"

    encoded = categorical.fit_transform(
        runner.TableTensor.from_pandas(
            pd.DataFrame(
                {
                    "numerical": [2.0, 4.0],
                    "categorical": ["zebra", "apple"],
                }
            ),
            stypes={
                "numerical": "numerical",
                "categorical": "categorical",
            },
        )
    )
    assert encoded.columns[runner.Stype.numerical] == (
        "categorical",
        "numerical",
    )
    runner.torch.testing.assert_close(
        encoded.numerical,
        runner.torch.tensor([[1.0, 2.0], [0.0, 4.0]]),
    )

    standard = recipe.features.steps[3]
    assert isinstance(standard, StandardScale)
    assert standard.epsilon == 1e-6
    sigma = recipe.features.steps[6]
    assert isinstance(sigma, SigmaClip)
    assert sigma.threshold == 4.0

    table = runner.TableTensor.from_tensor(
        runner.torch.tensor([[-200.0, 200.0]])
    )
    clipped = recipe.features.steps[4].transform(table)
    runner.torch.testing.assert_close(
        clipped.numerical,
        runner.torch.tensor([[-100.0, 100.0]]),
    )


def test_timer_synchronizes_before_and_after_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    times = iter([1_000_000_000, 1_250_000_000])
    monkeypatch.setattr(
        runner,
        "_sync_cuda",
        lambda _num_gpus: events.append("sync"),
    )
    monkeypatch.setattr(
        runner.time,
        "perf_counter_ns",
        lambda: next(times),
    )

    value, elapsed = runner._timed_call(
        lambda: events.append("call") or "result",
        num_gpus=1,
    )

    assert value == "result"
    assert elapsed == 0.25
    assert events == ["sync", "call", "sync"]


def test_execution_order_alternates() -> None:
    assert runner.execution_order(0, 0) == (
        runner.ORIGINAL,
        runner.SDM,
    )
    assert runner.execution_order(0, 1) == (
        runner.SDM,
        runner.ORIGINAL,
    )
    assert runner.execution_order(1, 0) == (
        runner.SDM,
        runner.ORIGINAL,
    )


def test_local_schedule_discards_warmups_and_records_every_call() -> None:
    calls: list[dict[str, Any]] = []

    def record(spec: Mapping[str, Any]) -> dict[str, Any]:
        calls.append(dict(spec))
        return _fake_worker(spec)

    trial_rows, inference_rows = runner.run_local_benchmarks(
        _base_spec(),
        warmup_pairs=1,
        trials=2,
        worker_runner=record,
    )

    assert len(calls) == 12
    assert (
        len(
            [call for call in calls if str(call["trial"]).startswith("warmup")]
        )
        == 4
    )
    assert len(trial_rows) == 8
    assert len(inference_rows) == 24
    assert all(isinstance(row["trial"], int) for row in trial_rows)
    assert {row["comparison_group"] for row in trial_rows} == set(
        runner.LOCAL_PROFILES
    )
    assert all(
        row["eligible_for_fair_speedup"]
        is (row["comparison_group"] == runner.MATCHED_PARITY)
        for row in trial_rows
    )


def test_aggregation_limits_speedups_to_matched_parity() -> None:
    trial_rows, _ = _local_results()
    summary = runner.aggregate_local_results(
        trial_rows,
        trials=2,
        inference_repeats=3,
    )

    assert len(summary) == 4
    matched = [
        row
        for row in summary
        if row["comparison_group"] == runner.MATCHED_PARITY
    ]
    native = [
        row
        for row in summary
        if row["comparison_group"] == runner.LOCAL_NATIVE_DEFAULT
    ]
    original = next(
        row for row in matched if row["implementation"] == runner.ORIGINAL
    )
    sdm = next(row for row in matched if row["implementation"] == runner.SDM)
    assert original["fit_time_s_mean"] == 2.0
    assert original["fit_time_s_std"] == 0.0
    assert original["fit_speedup_vs_original"] == 1.0
    assert original["fit_speedup_vs_original_mean"] == 1.0
    assert original["fit_speedup_vs_original_std"] == 0.0
    assert sdm["fit_time_s_mean"] == 1.0
    assert sdm["fit_time_s_std"] == 0.0
    assert sdm["fit_speedup_vs_original"] == 2.0
    assert sdm["fit_speedup_vs_original_mean"] == 2.0
    assert sdm["fit_speedup_vs_original_std"] == 0.0
    assert all(
        row["fit_speedup_vs_original"] is None
        and row["inference_speedup_vs_original"] is None
        and row["eligible_for_fair_speedup"] is False
        for row in native
    )


def test_aggregation_uses_sample_std_and_paired_trial_speedups() -> None:
    trial_rows, _ = _local_results(trials=3)
    original_fit = [2.0, 4.0, 8.0]
    sdm_fit = [1.0, 1.0, 2.0]
    for row in trial_rows:
        if row["comparison_group"] != runner.MATCHED_PARITY:
            continue
        trial = row["trial"]
        assert isinstance(trial, int)
        values = (
            original_fit
            if row["implementation"] == runner.ORIGINAL
            else sdm_fit
        )
        row["fit_time_s"] = values[trial]

    summary = runner.aggregate_local_results(
        trial_rows,
        trials=3,
        inference_repeats=3,
    )
    matched = [
        row
        for row in summary
        if row["comparison_group"] == runner.MATCHED_PARITY
        and row["implementation"] == runner.SDM
    ]
    sdm = next(iter(matched))
    assert sdm["fit_time_s_mean"] == pytest.approx(4 / 3)
    assert sdm["fit_time_s_std"] == pytest.approx(0.5773502692)
    assert sdm["fit_time_s"] == 1.0
    assert sdm["fit_time_s_q1"] == 1.0
    assert sdm["fit_time_s_q3"] == 1.5
    assert sdm["fit_speedup_vs_original_mean"] == pytest.approx(10 / 3)
    assert sdm["fit_speedup_vs_original_std"] == pytest.approx(1.1547005384)
    assert sdm["fit_speedup_vs_original"] == 4.0
    assert sdm["fit_speedup_vs_original_q1"] == 3.0
    assert sdm["fit_speedup_vs_original_q3"] == 4.0


def test_single_trial_sample_std_is_not_claimed() -> None:
    trial_rows, _ = _local_results(trials=1)
    summary = runner.aggregate_local_results(
        trial_rows,
        trials=1,
        inference_repeats=3,
    )
    assert all(row["fit_time_s_std"] is None for row in summary)
    matched = [
        row
        for row in summary
        if row["comparison_group"] == runner.MATCHED_PARITY
    ]
    assert all(row["fit_speedup_vs_original_std"] is None for row in matched)


def test_historical_baseline_is_external_and_not_comparable() -> None:
    historical = _historical_summary()

    assert historical["comparison_group"] == runner.HISTORICAL_TABARENA
    assert historical["execution_source"] == "tabarena_cached"
    assert historical["timing_comparability"] == "different_environment"
    assert historical["eligible_for_fair_speedup"] is False
    assert historical["fit_speedup_vs_original"] is None
    assert historical["historical_suite"] == "tabarena-test-suite"


@pytest.mark.parametrize(
    ("frame", "match"),
    [
        (_historical_frame().iloc[0:0], "got 0"),
        (_historical_frame(duplicate=True), "got 2"),
    ],
)
def test_historical_selection_rejects_missing_or_ambiguous_rows(
    frame: pd.DataFrame,
    match: str,
) -> None:
    with pytest.raises(RuntimeError, match=match):
        runner.select_historical_baseline(
            frame,
            dataset=runner.DEFAULT_DATASET,
            tabarena_split=0,
            task_id=runner.DEFAULT_TASK_ID,
            repeat=0,
            fold=0,
            sample=0,
            suite="tabarena-test-suite",
            config="TabICLv2_c1_BAG_L1",
        )


def test_report_uses_three_disjoint_tables() -> None:
    trial_rows, _ = _local_results()
    summary = runner.aggregate_local_results(
        trial_rows,
        trials=2,
        inference_repeats=3,
    )
    summary.append(_historical_summary())

    report = runner.render_report(
        summary,
        checkpoint_sha256="a" * 64,
        warmup_pairs=1,
        trials=2,
        inference_repeats=3,
    )

    matched_heading = "## 1. Matched parity"
    native_heading = "## 2. Local native defaults"
    historical_heading = "## 3. Historical TabArena baseline"
    assert report.index(matched_heading) < report.index(native_heading)
    assert report.index(native_heading) < report.index(historical_heading)
    assert "only rows eligible for fair speedup claims" in report
    assert "must not be treated as a parity comparison" in report
    assert "excluded from every local speedup" in report
    assert "2 measured trials after 1 discarded warm-up pair" in report
    assert "mean ± sample SD; median [Q1, Q3]" in report
    assert "per same-index trial" in report
    assert "2.000000 ± 0.000000; 2.000000 [2.000000, 2.000000]" in report
    native_section = report[
        report.index(native_heading) : report.index(historical_heading)
    ]
    assert "Fit speedup" not in native_section
    historical_section = report[report.index(historical_heading) :]
    assert "Fit speedup" not in historical_section
    assert "not recorded by TabArena, uploaded, or shared" in report


def test_worker_result_rejects_wrong_inference_count() -> None:
    spec = {
        **_base_spec(),
        "profile": runner.MATCHED_PARITY,
        "implementation": runner.ORIGINAL,
        "trial": 0,
        "execution_order": 0,
    }
    result = _fake_worker(spec)
    result["inference_times_s"] = [0.1]

    with pytest.raises(RuntimeError, match="wrong number"):
        runner._validate_worker_result(result, spec=spec)


def test_worker_subprocess_failure_is_propagated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = {
        **_base_spec(),
        "profile": runner.MATCHED_PARITY,
        "implementation": runner.ORIGINAL,
        "trial": 0,
        "execution_order": 0,
    }

    def fail(
        *_args: object, **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=[],
            returncode=2,
            stdout="worker stdout",
            stderr="worker stderr",
        )

    monkeypatch.setattr(runner.subprocess, "run", fail)
    with pytest.raises(RuntimeError, match="worker stderr"):
        runner.run_worker_subprocess(
            spec,
            scratch_dir=tmp_path,
            worker_timeout_s=1.0,
        )


def test_main_generates_classified_local_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "checkpoint.ckpt"
    checkpoint.write_bytes(b"benchmark checkpoint")
    checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    output_dir = tmp_path / "output"

    monkeypatch.setattr(
        runner,
        "run_worker_subprocess",
        lambda spec, **_kwargs: _fake_worker(dict(spec)),
    )
    monkeypatch.setattr(
        runner,
        "load_historical_baseline",
        lambda **_kwargs: _historical_summary(),
    )
    monkeypatch.setattr(runner, "_git_commit", lambda _path: "sdm-commit")
    monkeypatch.setattr(runner, "_tabarena_commit", lambda: "ta-commit")
    monkeypatch.setattr(
        runner,
        "environment_manifest",
        lambda **_kwargs: {"test": True},
    )

    assert (
        runner.main(
            [
                "--checkpoint-path",
                str(checkpoint),
                "--checkpoint-sha256",
                checkpoint_sha256,
                "--checkpoint-repository",
                "caller/repository",
                "--checkpoint-revision",
                "caller-revision",
                "--output-dir",
                str(output_dir),
                "--num-cpus",
                "1",
                "--num-gpus",
                "0",
                "--warmup-pairs",
                "1",
                "--trials",
                "2",
                "--inference-repeats",
                "3",
            ]
        )
        == 0
    )

    trials = pd.read_csv(output_dir / runner.TRIALS_FILENAME)
    inference = pd.read_csv(output_dir / runner.INFERENCE_FILENAME)
    summary = pd.read_csv(output_dir / runner.SUMMARY_FILENAME)
    report = (output_dir / runner.REPORT_FILENAME).read_text()
    manifest = json.loads((output_dir / runner.MANIFEST_FILENAME).read_text())

    assert len(trials) == 8
    assert len(inference) == 24
    assert len(summary) == 5
    assert (
        summary["comparison_group"].tolist().count(runner.HISTORICAL_TABARENA)
        == 1
    )
    matched = summary[summary["comparison_group"] == runner.MATCHED_PARITY]
    local = summary[summary["execution_source"] == "local_measured"]
    assert local["fit_time_s_mean"].notna().all()
    assert local["fit_time_s_std"].notna().all()
    assert matched["fit_speedup_vs_original"].notna().all()
    assert matched["fit_speedup_vs_original_mean"].notna().all()
    assert matched["fit_speedup_vs_original_std"].notna().all()
    not_matched = summary[summary["comparison_group"] != runner.MATCHED_PARITY]
    assert not_matched["fit_speedup_vs_original"].isna().all()
    assert not_matched["fit_speedup_vs_original_mean"].isna().all()
    assert "## 1. Matched parity" in report
    assert "## 2. Local native defaults" in report
    assert "## 3. Historical TabArena baseline" in report
    assert manifest["status"] == "completed"
    assert manifest["checkpoint"]["repository"] == "caller/repository"
    assert set(manifest["artifacts"]) == {
        runner.TRIALS_FILENAME,
        runner.INFERENCE_FILENAME,
        runner.SUMMARY_FILENAME,
        runner.REPORT_FILENAME,
    }
    assert not any(
        path.name.startswith(".workers-") for path in output_dir.iterdir()
    )
