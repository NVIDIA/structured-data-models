"""Compare prediction entry points after one eager fit on CPU."""

# ruff: noqa: D103, T201, BLE001

import argparse
import copy
import hashlib
import json
import platform
import statistics
import subprocess
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from sdm import CategoricalTensor, TableTensor
from sdm.models import KumoTabular


def graph_stats() -> dict[str, int]:
    return dict(torch._dynamo.utils.counters["stats"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--task", choices=["classification", "regression"], required=True
    )
    parser.add_argument("--estimators", type=int, default=1)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repeats", type=int, default=9)
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "task": args.task,
        "estimators": args.estimators,
        "torch": torch.__version__,
        "source": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "platform": platform.platform(),
        "dtype": "float32",
        "device": "cpu",
        "threads": 1,
        "backend": "inductor",
        "dynamic": True,
        "fullgraph": False,
        "context_rows": 32,
        "data_sha256": hashlib.sha256(
            Path(args.data).read_bytes()
        ).hexdigest(),
        "cold": [],
        "timings": [],
    }

    def save() -> None:
        output.write_text(json.dumps(result, indent=2) + "\n")

    torch.set_num_threads(1)
    torch.manual_seed(123)
    data = np.load(args.data)
    cx = TableTensor.from_tensor(
        torch.tensor(data["x"][data["train_ids"][:32]])
    )
    target = torch.tensor(data["y"][data["train_ids"][:32]]).view(-1, 1)
    cy = TableTensor.from_tensor(
        CategoricalTensor.from_tensor(target.long())
        if args.task == "classification"
        else target.float()
    )
    row_counts = [32, min(128, len(data["validation_ids"]))]
    queries = {
        rows: TableTensor.from_tensor(
            torch.tensor(data["x"][data["validation_ids"][:rows]])
        )
        for rows in row_counts
    }
    result["query_rows"] = row_counts
    model = KumoTabular(
        task=args.task, size="small", pretrained=False, device="cpu"
    )
    model.models[args.task].load_state_dict(
        torch.load(args.checkpoint, weights_only=True, map_location="cpu"),
        strict=True,
    )
    with torch.inference_mode():
        model.fit(
            cx,
            cy,
            generator=torch.Generator().manual_seed(123),
            num_estimators=args.estimators,
        )
        expected = {
            rows: model.predict(query).numerical.clone()
            for rows, query in queries.items()
        }
        # Identical fitted weights, processor state, and attention caches.
        inner_model = copy.deepcopy(model)
        public_model = copy.deepcopy(model)
        for module in inner_model.models.values():
            module.compile(backend="inductor", fullgraph=False, dynamic=True)
        calls = {
            "eager": model.predict,
            "inner": inner_model.predict,
            "predict": torch.compile(
                public_model.predict,
                backend="inductor",
                fullgraph=False,
                dynamic=True,
            ),
        }
        torch._dynamo.utils.counters.clear()
        successful = ["eager"]
        cold_outputs = {}
        for name in ["inner", "predict"]:
            try:
                for rows in [row_counts[0], row_counts[1], row_counts[0]]:
                    before = graph_stats()
                    start = time.perf_counter()
                    actual = calls[name](queries[rows]).numerical
                    elapsed = time.perf_counter() - start
                    ref = expected[rows]
                    result["cold"].append(
                        {
                            "path": name,
                            "rows": rows,
                            "seconds": elapsed,
                            "before": before,
                            "after": graph_stats(),
                            "max_abs_error": (actual - ref).abs().max().item(),
                            "parity": bool(
                                torch.allclose(
                                    actual, ref, atol=1e-5, rtol=1e-4
                                )
                            ),
                        }
                    )
                    entry = result["cold"][-1]
                    close = torch.isclose(actual, ref, atol=1e-5, rtol=1e-4)
                    bad = (~close).nonzero()
                    entry["shape"] = list(actual.shape)
                    entry["failed_count"] = len(bad)
                    entry["total_count"] = actual.numel()
                    entry["failed_values"] = [
                        {
                            "index": index.tolist(),
                            "eager": ref[tuple(index)].item(),
                            "actual": actual[tuple(index)].item(),
                            "abs_error": (actual - ref)[tuple(index)]
                            .abs()
                            .item(),
                            "threshold": (
                                1e-5 + 1e-4 * ref[tuple(index)].abs()
                            ).item(),
                        }
                        for index in bad
                    ]
                    if args.task == "regression":
                        labels = torch.tensor(
                            data["y"][data["validation_ids"][:rows]],
                            dtype=actual.dtype,
                        )
                        median_index = (actual.size(-1) - 1) // 2
                        entry["median_quantile_column"] = median_index
                        entry["metrics"] = {}
                        for arm, values in [
                            ("eager", ref),
                            ("compiled", actual),
                        ]:
                            residual = values[:, median_index] - labels
                            entry["metrics"][arm] = {
                                "rmse": residual.square().mean().sqrt().item(),
                                "mae": residual.abs().mean().item(),
                            }
                    if name == "predict":
                        inner = cold_outputs[("inner", rows)]
                        entry["public_minus_inner_max_abs"] = (
                            (actual - inner).abs().max().item()
                        )
                        entry["public_inner_equal"] = bool(
                            torch.equal(actual, inner)
                        )
                    cold_outputs[(name, rows)] = actual.clone()
                    save()
                successful.append(name)
            except Exception as error:
                result.setdefault("errors", {})[name] = {
                    "type": type(error).__name__,
                    "message": str(error),
                    "traceback": traceback.format_exc(),
                }
                save()
        # Warm both shapes and rotate timing order across entry points.
        for rows in row_counts:
            for _ in range(2):
                for name in successful:
                    calls[name](queries[rows])
            for repeat in range(args.repeats):
                order = (
                    successful[repeat % len(successful) :]
                    + successful[: repeat % len(successful)]
                )
                for name in order:
                    before = graph_stats().get("unique_graphs", 0)
                    start = time.perf_counter()
                    actual = calls[name](queries[rows]).numerical
                    elapsed = time.perf_counter() - start
                    result["timings"].append(
                        {
                            "path": name,
                            "rows": rows,
                            "repeat": repeat,
                            "seconds": elapsed,
                            "new_graphs": graph_stats().get("unique_graphs", 0)
                            - before,
                            "max_abs_error": (actual - expected[rows])
                            .abs()
                            .max()
                            .item(),
                            "parity": bool(
                                torch.allclose(
                                    actual,
                                    expected[rows],
                                    atol=1e-5,
                                    rtol=1e-4,
                                )
                            ),
                        }
                    )
                    save()
        result["summary"] = [
            {
                "path": name,
                "rows": rows,
                "median_ms": 1000
                * statistics.median(
                    [
                        v["seconds"]
                        for v in result["timings"]
                        if v["path"] == name and v["rows"] == rows
                    ]
                ),
                "min_ms": 1000
                * min(
                    [
                        v["seconds"]
                        for v in result["timings"]
                        if v["path"] == name and v["rows"] == rows
                    ]
                ),
                "max_ms": 1000
                * max(
                    [
                        v["seconds"]
                        for v in result["timings"]
                        if v["path"] == name and v["rows"] == rows
                    ]
                ),
            }
            for rows in row_counts
            for name in successful
        ]
        result["final_graph_stats"] = graph_stats()
        result["graph_breaks"] = dict(
            torch._dynamo.utils.counters["graph_break"]
        )
        result["all_parity"] = all(
            sample["parity"] for sample in result["cold"] + result["timings"]
        )
        result["status"] = "complete"
        save()
        print(json.dumps(result["summary"]), flush=True)


if __name__ == "__main__":
    main()
