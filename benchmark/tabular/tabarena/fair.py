# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run one pinned official TabArena split with public model wrappers."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import tabarena
import torch
from tabarena.benchmark.exec_models.autogluon import AGSingleBagWrapper
from tabarena.benchmark.experiment import TabArenaV0pt1ExperimentBundle
from tabarena.caching import CacheConfig
from tabarena.contexts import TabArenaContext
from tabarena.models.utils import get_model_info_from_name
from tabarena.utils.config_utils import ConfigGenerator
from tabarena.utils.pickle_utils import load_pickle

import benchmark.tabular.model as adapter


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(root: Path, manifest_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text())
    for name, expected in manifest["files"].items():
        if digest(root / name) != expected:
            raise ValueError(f"Frozen source changed: {root / name}")
    for name, expected in manifest.get("symlinks", {}).items():
        if os.readlink(root / name) != expected:
            raise ValueError(f"Frozen symlink changed: {root / name}")
    identity = {
        key: value
        for key, value in manifest.items()
        if key not in {"files", "symlinks"}
    }
    identity["manifest_sha256"] = digest(manifest_path)
    return identity


def write_manifest() -> None:
    parser = argparse.ArgumentParser(
        description="Record a Git checkout's current tracked source bytes."
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(sys.argv[2:])
    root = args.root.resolve()

    def git(*arguments: str) -> bytes:
        return subprocess.check_output(["git", "-C", str(root), *arguments])

    files, symlinks = {}, {}
    for name in git("ls-files", "-z").decode().split("\0"):
        if not name:
            continue
        path = root / name
        if path.is_symlink():
            symlinks[name] = os.readlink(path)
        elif path.is_file():
            files[name] = digest(path)
    diff = git("diff", "--binary", "HEAD")
    manifest = {
        "commit": git("rev-parse", "HEAD").decode().strip(),
        "dirty_tracked_source": bool(diff),
        "git_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "files": files,
        "symlinks": symlinks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {
                "manifest": str(args.output),
                "commit": manifest["commit"],
                "files": len(files),
            }
        )
    )


def main() -> None:
    if sys.argv[1:2] == ["manifest"]:
        write_manifest()
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=["kumo", "kumo-batched", "causilo", "tabpfn-fast"],
        required=True,
    )
    parser.add_argument("--dataset")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--refit-folds", choices=["no"], default="no")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--tabarena-root", type=Path, required=True)
    parser.add_argument("--tabarena-manifest", type=Path, required=True)
    parser.add_argument("--openml-cache", type=Path, required=True)
    parser.add_argument("--tabarena-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-cpus", type=int, default=8)
    parser.add_argument("--min-free-gib", type=float, default=128)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--export-grid", type=Path)
    args = parser.parse_args()
    identity = {
        "sdm": verify(args.source_root, args.source_manifest),
        "tabarena": verify(args.tabarena_root, args.tabarena_manifest),
        "harness_sha256": digest(Path(__file__)),
    }

    if (
        not Path(adapter.__file__)
        .resolve()
        .is_relative_to(args.source_root.resolve())
    ):
        raise ValueError(f"Wrong SDM adapter import: {adapter.__file__}")
    if args.model == "kumo-batched" and not callable(
        getattr(adapter.SDMKumoTabularModel, "_estimator_batch_size", None)
    ):
        raise ValueError(
            "The selected source does not support estimator batching"
        )
    if (
        not Path(tabarena.__file__)
        .resolve()
        .is_relative_to(args.tabarena_root.resolve())
    ):
        raise ValueError(f"Wrong TabArena import: {tabarena.__file__}")
    CacheConfig(openml=args.openml_cache, tabarena=args.tabarena_cache).apply()
    torch.set_num_threads(args.num_cpus)
    output = args.output
    refit = args.refit_folds == "yes"
    protocol = {
        "outer_experiments": False,
        "refit_folds": refit,
        "num_cpus": args.num_cpus,
        "num_gpus": 1,
        "sequential_local_fold_fitting": True,
    }
    hyperparameters = {"ag_args_ensemble": {"refit_folds": refit}}
    if args.model in {"kumo", "kumo-batched"}:
        hyperparameters.update(
            {
                "num_estimators": 8,
                "estimator_batch_size": "auto"
                if args.model == "kumo-batched"
                else 1,
                "max_context_size": None,
                "max_columns": None,
                "kv_cache": False,
                "ag_args_fit": {"share_pretrained_weights": True},
            }
        )
        model_config = adapter.MODEL_CONFIGS["kumo-tabular"]
        generator = ConfigGenerator(
            search_space={},
            model_cls=model_config.model_cls,
            manual_configs=[hyperparameters],
        )
        models = [(generator, 0)]
    else:
        registry_name = {
            "causilo": "Causilo",
            "tabpfn-fast": "TabPFN-3.5-Fast",
        }[args.model]
        info = get_model_info_from_name(registry_name)
        generator = copy.deepcopy(info.search_space)
        if len(generator.manual_configs) != 1:
            raise ValueError(
                "Expected exactly one official default configuration"
            )
        hyperparameters = generator.manual_configs[0]
        hyperparameters.setdefault("ag_args_ensemble", {})["refit_folds"] = (
            refit
        )
        models = [(generator, 0)]
    experiments = TabArenaV0pt1ExperimentBundle(
        models=models,
        outer_experiments=False,
        sequential_local_fold_fitting=True,
        model_artifacts_base_path=str(output / "models"),
    ).build_experiments(num_cpus=args.num_cpus, num_gpus=1)
    context = TabArenaContext(backend="native")
    jobs = context.build_jobs(
        experiments=experiments,
        dataset_names=[args.dataset] if args.dataset else None,
    )
    if args.export_grid:
        records = [
            {
                "dataset": job.task.dataset,
                "fold": job.task.fold,
                "repeat": job.task.repeat,
                "job": job.to_dict(),
            }
            for job in jobs
        ]
        args.export_grid.write_text(
            json.dumps(records, indent=2, default=str) + "\n"
        )
        print(
            json.dumps(
                {
                    "jobs": len(jobs),
                    "datasets": len({j.task.dataset for j in jobs}),
                    "cuda_initialized": torch.cuda.is_initialized(),
                    "identity": identity,
                }
            )
        )
        return
    jobs = [
        job
        for job in jobs
        if job.task.fold == args.fold and job.task.repeat == args.repeat
    ]
    if args.dataset is None or len(jobs) != 1:
        raise ValueError(
            f"Specify one dataset/repeat/fold; found {len(jobs)} jobs"
        )
    manifest = {
        "model": args.model,
        "identity": identity,
        "protocol": protocol,
        "hyperparameters": hyperparameters,
        "job": jobs[0].to_dict(),
    }
    if args.plan_only:
        print(json.dumps(manifest, indent=2, default=str))
        return
    output.mkdir(parents=True, exist_ok=True)
    if list(output.rglob("results.pkl")):
        raise ValueError("Use a new output directory for every attempt")
    if shutil.disk_usage(output).free < args.min_free_gib * 2**30:
        raise RuntimeError("Insufficient scratch space for this run")
    (output / "benchmark-manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str) + "\n"
    )
    result_root = (
        output / "results" / args.model / ("refit" if refit else "eight-folds")
    )
    observed = {}
    original_cleanup = AGSingleBagWrapper.cleanup

    def observed_cleanup(wrapper: AGSingleBagWrapper) -> None:
        # Native fit/predict timers and result assembly have finished.
        # Inspect metadata only; preserve predictions and model state.
        try:
            bag = wrapper._load_model()
            children = [
                bag.load_child(child) if isinstance(child, str) else child
                for child in bag.models
            ]
            observed["children"] = []
            for child in children:
                served = child.model
                entry = {
                    "wrapper_class": (
                        f"{type(child).__module__}.{type(child).__qualname__}"
                    ),
                    "native_class": (
                        f"{type(served).__module__}."
                        f"{type(served).__qualname__}"
                    ),
                    "wrapper_params": child._get_model_params(),
                    "wrapper_autocast_dtype": str(
                        getattr(child, "autocast_dtype", None)
                    ),
                    "context_shape_before_recipe": getattr(
                        child, "_context_shape", None
                    ),
                    "subsampled_context": getattr(
                        child, "_expand_query", None
                    ),
                    "internal_estimators": getattr(
                        child, "_num_estimators", None
                    ),
                    "native_params": served.get_params(deep=False)
                    if hasattr(served, "get_params")
                    else None,
                }
                contexts = getattr(child, "_contexts", None)
                if contexts is not None:
                    entry["context_count"] = len(contexts)
                    entry["context_shapes"] = [
                        list(context.x.shape) for context in contexts
                    ]
                if hasattr(served, "parameters"):
                    entry["parameter_dtypes"] = sorted(
                        {str(p.dtype) for p in served.parameters()}
                    )
                engine = getattr(served, "_engine", None)
                if engine is not None:
                    entry["parameter_dtypes"] = sorted(
                        {str(p.dtype) for p in engine.model.parameters()}
                    )
                    entry["engine_device"] = str(engine.device)
                observed["children"].append(entry)
            observed["bag_params"] = bag.params
            observed["scope"] = (
                "Metadata inspection after official timers, "
                "immediately before native cleanup"
            )
            observed["prediction_query_chunks"] = None
            observed["routing_scope"] = (
                "Configured policy and fitted context only; actual query "
                "chunks and member groups are not instrumented"
            )
        except Exception as error:  # noqa: BLE001
            # Preserve native cleanup even when metadata inspection fails.
            observed["inspection_error"] = repr(error)
        finally:
            original_cleanup(wrapper)

    with patch.object(AGSingleBagWrapper, "cleanup", observed_cleanup):
        context.run_jobs(jobs=jobs, expname=result_root, register=False)
    (output / "fitted-settings.json").write_text(
        json.dumps(observed, indent=2, default=str) + "\n"
    )
    paths = list(result_root.rglob("results.pkl"))
    if len(paths) != 1:
        raise RuntimeError(f"Expected one official result, found {len(paths)}")
    result = load_pickle(paths[0])
    validation = result["validation_protocol"]
    assert validation["key"] == "8x1", validation
    assert validation["bag_params"]["refit_folds"] is refit, validation
    assert validation["num_child_models"] == (1 if refit else 8), validation
    assert math.isfinite(result["metric_error"])
    assert math.isfinite(result["metric_error_val"])
    assert result["simulation_artifacts"]["pred_proba_dict_val"]
    assert "inspection_error" not in observed, observed
    assert len(observed["children"]) == (1 if refit else 8), observed
    assert not list((output / "models").rglob("model.pkl")), (
        "Native cleanup left model artifacts"
    )
    verify(args.source_root, args.source_manifest)
    verify(args.tabarena_root, args.tabarena_manifest)
    summary = {
        "status": "completed",
        "model": args.model,
        "dataset": args.dataset,
        "fold": args.fold,
        "repeat": args.repeat,
        "identity": identity,
        "protocol": protocol,
        "result_file": str(paths[0].relative_to(output)),
        "result_sha256": digest(paths[0]),
        "validation_protocol": validation,
        "metrics": {
            key: result[key]
            for key in [
                "metric",
                "metric_error",
                "metric_error_val",
                "time_train_s",
                "time_infer_s",
                "memory_usage",
            ]
        },
        "warmup_report": result["experiment_metadata"]["warmup_report"],
        "time_warmup_s": result["experiment_metadata"]["time_warmup_s"],
        "timing_audit": result.get(
            "timing_audit", result["experiment_metadata"].get("timing_audit")
        ),
        "fitted_settings_file": "fitted-settings.json",
        "fitted_settings_sha256": digest(output / "fitted-settings.json"),
        "native_cleanup_complete": True,
    }
    (output / "benchmark-result.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n"
    )
    print(json.dumps(summary, default=str))


if __name__ == "__main__":
    main()
