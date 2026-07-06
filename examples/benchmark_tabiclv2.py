"""Inference acceleration benchmark for :class:`TabICLv2`.

Sweeps NVIDIA-recommended inference configurations (TF32, bf16, SDPA
backend pinning, ``torch.compile`` modes) over representative in-context
learning workloads and reports wall-clock latency, table-stream cost
(amortized over a stream of fresh table shapes, including recompiles),
peak memory, and accuracy versus an fp32 reference.

Each cell runs in a fresh subprocess so that inductor caches, CUDA-graph
pools, and allocator state cannot leak between configurations::

    python examples/benchmark_tabiclv2.py --out bench.json --budget-minutes 110
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from typing import Any

import torch
from sdm.models import TabICLv2
from torch import Tensor
from torch.nn.attention import SDPBackend, sdpa_kernel

CONFIGS = {
    # name: (precision, sdpa_priority, compile_kwargs)
    "c0-fp32": ("fp32", False, None),
    "c1-tf32": ("tf32", False, None),
    "c2-bf16-autocast": ("bf16-autocast", False, None),
    "c3-bf16-full": ("bf16-full", False, None),
    "c4-bf16-cudnn-sdpa": ("bf16-full", True, None),
    "c5-compile": ("bf16-full", True, {"fullgraph": True}),
    "c6-compile-dynamic": (
        "bf16-full",
        True,
        {"fullgraph": True, "dynamic": True},
    ),
    "c7-compile-ro": (
        "bf16-full",
        True,
        {"fullgraph": True, "dynamic": True, "mode": "reduce-overhead"},
    ),
    "c8-compile-ma": (
        "bf16-full",
        True,
        {
            "fullgraph": True,
            "dynamic": True,
            "mode": "max-autotune-no-cudagraphs",
        },
    ),
}

WORKLOADS = {
    # name: (batch_shape, num_rows, num_columns, num_train_rows)
    "small": ((), 256, 8, 192),
    "medium": ((), 2048, 32, 1536),
    "large": ((), 8192, 64, 6144),
    "batched": ((8,), 1024, 16, 768),
}

STREAM_LENGTH = 8  # Distinct table shapes per table-stream measurement.


def make_table(
    workload: str,
    task: str,
    seed: int,
    device: torch.device,
    jitter: int = 0,
) -> tuple[Tensor, Tensor]:
    """Generate a seeded, signal-bearing table for the given workload."""
    batch_shape, num_rows, num_columns, num_train = WORKLOADS[workload]
    if jitter:
        generator = torch.Generator().manual_seed(seed * 1_000 + jitter)
        num_rows += int(
            torch.randint(0, num_rows // 4, (), generator=generator)
        )
        num_columns += int(torch.randint(0, 4, (), generator=generator))
        num_train = (num_rows * 3) // 4
    torch.manual_seed(seed)

    x = torch.randn(*batch_shape, num_rows, num_columns, device=device)
    if task == "cls":
        num_classes = 4
        labels = torch.randint(
            0, num_classes, (*batch_shape, num_rows), device=device
        )
        # Shift a subset of columns per class so logit margins are
        # meaningful (pure-noise tables make top-1 agreement a coin flip).
        shift = torch.randn(num_classes, num_columns, device=device)
        x = x + 2.0 * shift[labels]
        y = labels[..., :num_train]
    else:
        weights = torch.randn(num_columns, device=device)
        target = x @ weights + 0.1 * torch.randn(
            *batch_shape, num_rows, device=device
        )
        y = target[..., :num_train]
    return x, y


def apply_precision(model: TabICLv2, precision: str) -> None:
    """Apply the requested precision mode to globals and the model."""
    if precision == "fp32":
        torch.backends.cuda.matmul.fp32_precision = "ieee"
    elif precision in ("tf32", "bf16-autocast"):
        torch.set_float32_matmul_precision("high")
    elif precision == "bf16-full":
        torch.set_float32_matmul_precision("high")
        model.to(torch.bfloat16)
    else:
        raise ValueError(f"unknown precision '{precision}'")


def cast_inputs(x: Tensor, y: Tensor, precision: str) -> tuple[Tensor, Tensor]:
    """Cast inputs to match the precision mode."""
    if precision == "bf16-full":
        x = x.to(torch.bfloat16)
        if y.is_floating_point():
            y = y.to(torch.bfloat16)
    return x, y


def timed_loop(fn: Any, min_seconds: float = 3.0) -> dict[str, float]:
    """Measure sync-bounded wall-clock latency of ``fn``."""
    times: list[float] = []
    while sum(times) < min_seconds or len(times) < 30:
        if len(times) >= 1000:
            break
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    times.sort()
    n = len(times)
    return {
        "p50_s": statistics.median(times),
        "iqr_s": times[(3 * n) // 4] - times[n // 4],
        "min_s": times[0],
        "iters": n,
    }


def accuracy_block(
    task: str,
    out: Tensor,
    ref: Tensor,
) -> dict[str, float | bool]:
    """Compare a configuration's output against the fp32 reference."""
    out = out.float()
    ref = ref.float()
    block: dict[str, float | bool] = {
        "max_abs_diff": (out - ref).abs().max().item()
    }
    if task == "cls":
        agree = (out.argmax(-1) == ref.argmax(-1)).float().mean().item()
        top2 = ref.topk(2, dim=-1).values
        margin = (top2[..., 0] - top2[..., 1]).median().item()
        mean_delta = (out - ref).abs().mean().item()
        block["top1_agreement"] = agree
        block["margin_ratio"] = mean_delta / max(margin, 1e-9)
        block["pass"] = agree >= 0.995 and block["margin_ratio"] < 0.1
    else:
        iqr = (ref[..., 988] - ref[..., 9]).abs().clamp(min=1e-9)
        rel = (out - ref).abs() / iqr.unsqueeze(-1)
        block["median_rel_err"] = rel.median().item()
        block["p99_rel_err"] = rel.flatten().quantile(0.99).item()
        block["pass"] = (
            block["median_rel_err"] < 1e-2 and block["p99_rel_err"] < 5e-2
        )
    return block


def run_cell(spec: dict[str, Any], workdir: str) -> dict[str, Any]:
    """Run one (config, workload, task, route) benchmark cell."""
    config, workload = spec["config"], spec["workload"]
    task, route = spec["task"], spec["route"]
    precision, pin_sdpa, compile_kwargs = CONFIGS[config]
    device = torch.device("cuda")
    result: dict[str, Any] = dict(spec)

    model = TabICLv2(pretrained=True, device=device)
    apply_precision(model, precision)

    contexts: list[Any] = []
    if pin_sdpa:
        contexts.append(
            sdpa_kernel(
                [
                    SDPBackend.CUDNN_ATTENTION,
                    SDPBackend.FLASH_ATTENTION,
                    SDPBackend.EFFICIENT_ATTENTION,
                    SDPBackend.MATH,
                ],
                set_priority=True,
            )
        )
    if precision == "bf16-autocast":
        contexts.append(torch.amp.autocast("cuda", torch.bfloat16))

    def call(x: Tensor, y: Tensor) -> Tensor:
        for ctx in contexts:
            ctx.__enter__()
        try:
            return model(x, y)
        finally:
            for ctx in reversed(contexts):
                ctx.__exit__(None, None, None)

    cold_compile_s = 0.0
    if compile_kwargs is not None:
        model.cls_model.compile(**compile_kwargs)
        model.reg_model.compile(**compile_kwargs)

    x, y = make_table(workload, task, seed=0, device=device)
    x, y = cast_inputs(x, y, precision)

    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    out = call(x, y).clone()
    torch.cuda.synchronize()
    cold_compile_s = time.perf_counter() - start
    result["cold_first_call_s"] = cold_compile_s

    if route == "oneshot":
        for _ in range(3):  # Warmup (post-compile).
            call(x, y)
        result.update(timed_loop(lambda: call(x, y)))

        # Table-stream: fresh shapes, including any recompiles.
        stream_start = time.perf_counter()
        for jitter in range(1, STREAM_LENGTH + 1):
            xs, ys = make_table(
                workload, task, seed=jitter, device=device, jitter=jitter
            )
            xs, ys = cast_inputs(xs, ys, precision)
            call(xs, ys)
        torch.cuda.synchronize()
        result["table_stream_per_table_s"] = (
            time.perf_counter() - stream_start
        ) / STREAM_LENGTH
    else:  # fit/predict route.
        num_train = WORKLOADS[workload][3]
        x_train, x_test = x[..., :num_train, :], x[..., num_train:, :]

        def fit_predict() -> Tensor:
            for ctx in contexts:
                ctx.__enter__()
            try:
                model.fit(x_train, y)
                pred = model.predict(x_test)
                model.clear()
                return pred
            finally:
                for ctx in reversed(contexts):
                    ctx.__exit__(None, None, None)

        def predict_only() -> Tensor:
            for ctx in contexts:
                ctx.__enter__()
            try:
                return model.predict(x_test)
            finally:
                for ctx in reversed(contexts):
                    ctx.__exit__(None, None, None)

        out = fit_predict().clone()
        fit_predict()  # Warmup both graphs.
        start = time.perf_counter()
        model.fit(x_train, y)
        torch.cuda.synchronize()
        result["fit_s"] = time.perf_counter() - start
        predict_only()
        result.update(timed_loop(predict_only))
        model.clear()

    result["peak_mem_mb"] = torch.cuda.max_memory_allocated() / 2**20

    # Accuracy vs the fp32 reference produced by the c0 cell.
    ref_path = os.path.join(workdir, f"ref-{workload}-{task}-{route}.pt")
    if config == "c0-fp32":
        torch.save(out.float().cpu(), ref_path)
        result["accuracy"] = {"pass": True, "is_reference": True}
    elif os.path.exists(ref_path):
        ref = torch.load(ref_path, map_location=device)
        result["accuracy"] = accuracy_block(task, out, ref)
    else:
        result["accuracy"] = {"pass": None, "missing_reference": True}
    return result


def spawn_cell(
    spec: dict[str, Any],
    out_path: str,
    workdir: str,
) -> dict[str, Any] | None:
    """Run one cell in an isolated subprocess and return its record."""
    env = dict(os.environ)
    env["TORCHINDUCTOR_CACHE_DIR"] = tempfile.mkdtemp(
        prefix=f"inductor-{spec['config']}-", dir=workdir
    )
    proc = subprocess.run(
        [
            sys.executable,
            os.path.abspath(__file__),
            "--run-cell",
            json.dumps(spec),
            "--out",
            out_path,
            "--workdir",
            workdir,
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if proc.returncode != 0:
        record = dict(spec)
        record["error"] = proc.stderr[-2000:]
        with open(out_path, "a") as handle:
            handle.write(json.dumps(record) + "\n")
        return record
    with open(out_path) as handle:
        return json.loads(handle.readlines()[-1])


def main() -> None:
    """Plan and execute the benchmark grid."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="bench.json")
    parser.add_argument("--budget-minutes", type=float, default=110.0)
    parser.add_argument("--run-cell", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--workdir", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.run_cell is not None:
        record = run_cell(json.loads(args.run_cell), args.workdir)
        with open(args.out, "a") as handle:
            handle.write(json.dumps(record) + "\n")
        return

    workdir = tempfile.mkdtemp(prefix="sdm-bench-")
    deadline = time.time() + args.budget_minutes * 60.0
    records: list[dict[str, Any]] = []

    def enqueue(cells: list[dict[str, Any]]) -> None:
        for spec in cells:
            if time.time() > deadline:
                spec = dict(spec)
                spec["skipped_walltime"] = True
                records.append(spec)
                with open(args.out, "a") as handle:
                    handle.write(json.dumps(spec) + "\n")
                continue
            record = spawn_cell(spec, args.out, workdir)
            if record is not None:
                records.append(record)
                print(format_record(record), flush=True)

    # Stage 1: full config grid on the size extremes (classification,
    # one-shot route). c0 first: it writes the accuracy reference.
    stage1 = [
        {
            "config": config,
            "workload": workload,
            "task": "cls",
            "route": "oneshot",
        }
        for workload in ("small", "large")
        for config in CONFIGS
    ]
    enqueue(stage1)

    # Sentinel: re-measure the baseline to detect clock/thermal drift.
    enqueue(
        [
            {
                "config": "c0-fp32",
                "workload": "large",
                "task": "cls",
                "route": "oneshot",
                "sentinel": True,
            }
        ]
    )

    # Stage 2: promote the top-2 non-baseline configs by large-table
    # latency to the remaining workloads, tasks, and the fit/predict route.
    scored = [
        r
        for r in records
        if r.get("workload") == "large"
        and r.get("p50_s")
        and r["config"] != "c0-fp32"
        and r.get("accuracy", {}).get("pass")
    ]
    top = [r["config"] for r in sorted(scored, key=lambda r: r["p50_s"])[:2]]
    stage2 = []
    for config in ["c0-fp32", *top]:
        for workload in ("medium", "batched"):
            stage2.append(
                {
                    "config": config,
                    "workload": workload,
                    "task": "cls",
                    "route": "oneshot",
                }
            )
        for workload in ("small", "large"):
            stage2.append(
                {
                    "config": config,
                    "workload": workload,
                    "task": "reg",
                    "route": "oneshot",
                }
            )
            stage2.append(
                {
                    "config": config,
                    "workload": workload,
                    "task": "cls",
                    "route": "fitpredict",
                }
            )
    enqueue(stage2)

    print("\n=== SUMMARY (per-table stream / steady-state p50) ===")
    for record in records:
        print(format_record(record))


def format_record(record: dict[str, Any]) -> str:
    """Format one benchmark record as a fixed-width table row."""
    if record.get("skipped_walltime"):
        return "{:<22} {:<8} {:<4} {:<10} SKIPPED (walltime)".format(
            record["config"],
            record["workload"],
            record["task"],
            record["route"],
        )
    if "error" in record:
        return "{:<22} {:<8} {:<4} {:<10} ERROR".format(
            record["config"],
            record["workload"],
            record["task"],
            record["route"],
        )
    accuracy = record.get("accuracy", {})
    return (
        "{:<22} {:<8} {:<4} {:<10} p50={:>8.2f}ms stream={:>8.2f}ms "
        "cold={:>6.1f}s mem={:>7.0f}MB pass={}".format(
            record["config"],
            record["workload"],
            record["task"],
            record["route"],
            1000 * record.get("p50_s", float("nan")),
            1000 * record.get("table_stream_per_table_s", float("nan")),
            record.get("cold_first_call_s", float("nan")),
            record.get("peak_mem_mb", float("nan")),
            accuracy.get("pass"),
        )
    )


if __name__ == "__main__":
    main()
