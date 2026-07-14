from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

pytest.importorskip("autogluon")
pytest.importorskip("tabarena")

import tabarena.contexts as tabarena_contexts
from examples.benchmarking import run_tabiclv2_tabarena_smoke as runner

pytestmark = pytest.mark.tabarena

SDM_METHOD = "[New] SDM-TabICLv2_c1"
ORIGINAL_METHOD = "TABICLV2 (default)"


def _raw_result() -> dict[str, Any]:
    return {
        "framework": "SDM-TabICLv2_c1",
        "problem_type": "regression",
        "metric": "rmse",
        "metric_error": 0.42,
        "time_train_s": 1.5,
        "time_infer_s": 0.25,
        "task_metadata": {
            "name": runner.DATASET,
            "fold": 0,
            "repeat": 0,
            "split_idx": 0,
        },
    }


def _all_results() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "method": ORIGINAL_METHOD,
                "dataset": runner.DATASET,
                "fold": 0,
                "metric": "rmse",
                "problem_type": "regression",
                "metric_error": 0.5,
                "time_train_s": 2.0,
                "time_infer_s": 0.1,
                "config_type": runner.ORIGINAL_CONFIG_TYPE,
                "method_subtype": runner.ORIGINAL_METHOD_SUBTYPE,
            },
            {
                "method": SDM_METHOD,
                "dataset": runner.DATASET,
                "fold": 0,
                "metric": "rmse",
                "problem_type": "regression",
                "metric_error": 0.42,
                "time_train_s": 1.5,
                "time_infer_s": 0.25,
                "config_type": "SDMTabICLv2Model",
                "method_subtype": "default",
            },
        ]
    )


def _new_results() -> pd.DataFrame:
    return _all_results().iloc[[1]].reset_index(drop=True)


class _FakeContext:
    def __init__(
        self,
        *,
        run_results: list[dict[str, Any]] | None = None,
        new_leaderboard: pd.DataFrame | None = None,
        new_results: pd.DataFrame | None = None,
    ) -> None:
        self.run_results = (
            [_raw_result()] if run_results is None else run_results
        )
        self.all_results = _all_results()
        self.new_leaderboard = (
            pd.DataFrame({"rank": [1]}, index=[SDM_METHOD])
            if new_leaderboard is None
            else new_leaderboard
        )
        self.new_results = (
            _new_results() if new_results is None else new_results
        )
        self.build_call: dict[str, Any] | None = None
        self.compare_calls: list[dict[str, Any]] = []

    def build_and_run_jobs(
        self,
        experiments: list[object],
        *,
        expname: str,
        build_kwargs: dict[str, Any],
        new_result_prefix: str,
        debug_mode: bool,
    ) -> list[dict[str, Any]]:
        self.build_call = {
            "experiments": experiments,
            "expname": expname,
            "build_kwargs": build_kwargs,
            "new_result_prefix": new_result_prefix,
            "debug_mode": debug_mode,
        }
        return self.run_results

    def compare(
        self,
        *,
        output_dir: Path | None,
        return_results: bool,
        new_methods_only: bool = False,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        self.compare_calls.append(
            {
                "output_dir": output_dir,
                "return_results": return_results,
                "new_methods_only": new_methods_only,
            }
        )
        assert return_results is True
        if new_methods_only:
            return self.new_leaderboard, self.new_results
        leaderboard = pd.DataFrame(
            {"rank": [1, 2]},
            index=[ORIGINAL_METHOD, SDM_METHOD],
        )
        return leaderboard, self.all_results

    @staticmethod
    def leaderboard_to_website_format(
        *,
        leaderboard: pd.DataFrame,
    ) -> pd.DataFrame:
        return pd.DataFrame({"method": leaderboard.index.tolist()})


def _runner_args(
    tmp_path: Path,
    *,
    debug_mode: bool = True,
) -> tuple[list[str], Path]:
    checkpoint = tmp_path / "checkpoint.ckpt"
    checkpoint.write_bytes(b"test checkpoint")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    output_dir = tmp_path / "output"
    args = [
        "--checkpoint-path",
        str(checkpoint),
        "--checkpoint-sha256",
        digest,
        "--checkpoint-repository",
        " caller/repository ",
        "--checkpoint-revision",
        " caller-revision ",
        "--num-cpus",
        "1",
        "--num-gpus",
        "0",
        "--output-dir",
        str(output_dir),
    ]
    if debug_mode:
        args.append("--debug-mode")
    return args, output_dir


def _install_context(
    monkeypatch: pytest.MonkeyPatch,
    context: _FakeContext,
) -> None:
    monkeypatch.setattr(
        tabarena_contexts,
        "TabArenaContext",
        lambda: context,
    )
    monkeypatch.setattr(runner, "_git_commit", lambda _path: "sdm-commit")
    monkeypatch.setattr(
        runner,
        "_tabarena_commit",
        lambda: "tabarena-commit",
    )
    monkeypatch.setattr(
        runner,
        "build_smoke_experiments",
        lambda **_kwargs: [object()],
    )


def test_checkpoint_provenance_options_are_required(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit):
        runner._parse_args(
            [
                "--checkpoint-path",
                "/tmp/checkpoint.ckpt",
                "--checkpoint-sha256",
                "0" * 64,
            ]
        )

    error = capsys.readouterr().err
    assert "--checkpoint-repository" in error
    assert "--checkpoint-revision" in error


@pytest.mark.parametrize(
    ("results", "match"),
    [
        ([], "got 0"),
        ([_raw_result(), _raw_result()], "got 2"),
    ],
)
def test_raw_run_validation_requires_exactly_one_result(
    results: list[dict[str, Any]],
    match: str,
) -> None:
    with pytest.raises(RuntimeError, match=match):
        runner._validate_run_results(results)


def test_raw_result_rejects_wrong_framework_and_nonfinite_metric() -> None:
    wrong_framework = _raw_result()
    wrong_framework["framework"] = "UnrelatedModel_c1"
    with pytest.raises(RuntimeError, match="SDM TabICLv2"):
        runner._validate_run_results([wrong_framework])

    nonfinite = _raw_result()
    nonfinite["metric_error"] = float("nan")
    with pytest.raises(RuntimeError, match="must be finite"):
        runner._validate_run_results([nonfinite])


def test_main_writes_caller_provenance_and_paired_comparison(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _FakeContext()
    _install_context(monkeypatch, context)
    args, output_dir = _runner_args(tmp_path)

    assert runner.main(args) == 0

    manifest = json.loads((output_dir / "manifest.json").read_text())
    comparison_path = output_dir / "evaluation" / runner.COMPARISON_FILENAME
    assert manifest["status"] == "completed"
    assert manifest["checkpoint"]["repository"] == "caller/repository"
    assert manifest["checkpoint"]["revision"] == "caller-revision"
    assert manifest["configuration"]["debug_mode"] is True
    assert manifest["comparison_path"] == str(comparison_path.resolve())

    comparison = pd.read_csv(comparison_path)
    assert comparison.columns.tolist() == [
        "implementation",
        "method",
        "dataset",
        "fold",
        "metric",
        "metric_error",
        "time_train_s",
        "time_infer_s",
    ]
    assert comparison["implementation"].tolist() == [
        "original_tabarena",
        "sdm",
    ]
    assert comparison["method"].tolist() == [ORIGINAL_METHOD, SDM_METHOD]

    assert context.build_call is not None
    assert context.build_call["new_result_prefix"] == runner.NEW_RESULT_PREFIX
    assert context.build_call["debug_mode"] is True
    assert [call["new_methods_only"] for call in context.compare_calls] == [
        False,
        True,
    ]
    assert context.compare_calls[0]["output_dir"] == (
        output_dir.resolve() / "evaluation"
    )
    assert context.compare_calls[1]["output_dir"] is None


def test_baseline_enriched_rows_cannot_satisfy_completion_check(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline_enriched = pd.DataFrame(
        {"rank": [1, 2]},
        index=[ORIGINAL_METHOD, SDM_METHOD],
    )
    context = _FakeContext(
        new_leaderboard=baseline_enriched,
        new_results=_all_results(),
    )
    _install_context(monkeypatch, context)
    args, output_dir = _runner_args(tmp_path)

    with pytest.raises(RuntimeError, match="exactly one new SDM"):
        runner.main(args)

    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert "got 2" in manifest["error"]


def test_no_raw_result_is_recorded_as_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _FakeContext(run_results=[])
    _install_context(monkeypatch, context)
    args, output_dir = _runner_args(tmp_path)

    with pytest.raises(RuntimeError, match="got 0"):
        runner.main(args)

    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert "exactly one SDM TabICLv2 result, got 0" in manifest["error"]
    assert context.compare_calls == []
    assert not (
        output_dir / "evaluation" / runner.COMPARISON_FILENAME
    ).exists()


def test_post_start_experiment_build_failure_is_recorded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _FakeContext()
    _install_context(monkeypatch, context)

    def fail_build(**_kwargs: object) -> list[object]:
        raise LookupError("experiment construction failed")

    monkeypatch.setattr(runner, "build_smoke_experiments", fail_build)
    args, output_dir = _runner_args(tmp_path, debug_mode=False)

    with pytest.raises(LookupError, match="construction failed"):
        runner.main(args)

    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["configuration"]["debug_mode"] is False
    assert manifest["error"].startswith("LookupError:")


def test_paired_comparison_rejects_ambiguous_original_result(
    tmp_path: Path,
) -> None:
    all_results = _all_results()
    all_results = pd.concat(
        [all_results, all_results.iloc[[0]]],
        ignore_index=True,
    )

    with pytest.raises(RuntimeError, match=r"original.*got 2"):
        runner._write_tabiclv2_comparison(
            tmp_path / runner.COMPARISON_FILENAME,
            all_results,
            _new_results(),
            expected_new_method=SDM_METHOD,
        )
