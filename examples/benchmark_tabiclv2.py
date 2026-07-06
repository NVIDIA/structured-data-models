r"""Inference acceleration benchmark for :class:`TabICLv2`.

Sweeps NVIDIA-recommended inference configurations over representative
in-context learning workloads and reports wall-clock latency,
table-stream cost (amortized over a stream of fresh table shapes,
including recompiles), peak memory, and accuracy versus an fp32
reference.

What to expect from each precision (measured on GB200, 8192x64-row
one-shot classification unless noted; see the PR for the full table):

* ``fp32`` (``ieee`` matmuls): the accuracy reference; slowest
  (103 ms).
* ``tf32``: 1.4x for fp32 pipelines with negligible drift; the safe
  default (``torch.set_float32_matmul_precision("high")``, 72 ms).
* ``bf16`` (full-cast): 4x and half the peak memory (26 ms), passing
  all accuracy gates; 6.5x when combined with
  ``compile(fullgraph=True, dynamic=True)`` (15.8 ms), and up to ~10x
  on small launch-bound tables with ``mode="reduce-overhead"``
  (1.3 ms). Static compilation reaches 7.1x steady-state but recompiles
  every new table shape (~5 s/table), so keep ``dynamic=True`` for
  in-context learning. The autocast variant of this recipe (as shipped
  in ``examples/tabiclv2.py``) measures 6.2x (16.6 ms) one-shot and
  ~9 ms fit/predict.
* ``fp8`` (Transformer Engine, per-tensor scaling): measured *slower*
  than bf16 here (27 ms large, 21 ms vs 11 ms small) - at this model's
  GEMM sizes (128-1536 channels) quantization overhead cancels the
  gain. Accuracy gates pass. Shapes are constrained: feature dims must
  be multiples of 16 and leading-dimension products multiples of 8, so
  arbitrary table sizes need padding.
* ``mxfp8`` (block-scaled fp8, Blackwell): the best of the 8-bit
  options (27 ms large) but still no win over bf16 at these GEMM
  sizes; same shape constraints as fp8.
* ``nvfp4`` (4-bit block-scaled, Blackwell): slowest of the family
  (33 ms large) and the only configuration to fail an accuracy gate:
  pooled over five seeded tables its median per-row logit shift is 26%
  of the decision margin (threshold 10%), with top-1 agreement at
  99.7%. Built for much larger GEMMs - not recommended for TabICLv2.

The fp8/mxfp8/nvfp4 configurations require the optional
``transformer_engine`` package (available in NVIDIA NGC containers) and
swap eligible ``torch.nn.Linear`` modules for ``te.Linear``.

Each cell runs in a fresh subprocess so that inductor caches, CUDA-graph
pools, and allocator state cannot leak between configurations::

    python examples/benchmark_tabiclv2.py --out bench.json \\
        --budget-minutes 110 --configs c0-fp32,c3-bf16-full,c10-mxfp8
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from contextlib import ExitStack
from typing import Any

import torch
from sdm.models import TabICLv2
from torch import Tensor
from torch.nn.attention import SDPBackend, sdpa_kernel

CONFIGS = {
    # name: (precision, sdpa_priority, compile_kwargs, te_recipe)
    "c0-fp32": ("fp32", False, None, None),
    "c1-tf32": ("tf32", False, None, None),
    "c2-bf16-autocast": ("bf16-autocast", False, None, None),
    "c3-bf16-full": ("bf16-full", False, None, None),
    "c4-bf16-cudnn-sdpa": ("bf16-full", True, None, None),
    "c5-compile": ("bf16-full", True, {"fullgraph": True}, None),
    "c6-compile-dynamic": (
        "bf16-full",
        True,
        {"fullgraph": True, "dynamic": True},
        None,
    ),
    "c7-compile-ro": (
        "bf16-full",
        True,
        {"fullgraph": True, "dynamic": True, "mode": "reduce-overhead"},
        None,
    ),
    "c8-compile-ma": (
        "bf16-full",
        True,
        {
            "fullgraph": True,
            "dynamic": True,
            "mode": "max-autotune-no-cudagraphs",
        },
        None,
    ),
    # The examples/tabiclv2.py recipe: autocast (TableTensor-friendly)
    # combined with dynamic fullgraph compilation.
    "c12-autocast-compile": (
        "bf16-autocast",
        False,
        {"fullgraph": True, "dynamic": True},
        None,
    ),
    # Transformer Engine low-precision GEMMs on a bf16 base model.
    "c9-fp8": ("bf16-full", True, None, "fp8"),
    "c10-mxfp8": ("bf16-full", True, None, "mxfp8"),
    "c11-nvfp4": ("bf16-full", True, None, "nvfp4"),
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


def load_transformer_engine() -> tuple[Any, Any]:
    """Import Transformer Engine lazily (optional heavy dependency)."""
    import importlib

    te = importlib.import_module("transformer_engine.pytorch")
    recipes = importlib.import_module("transformer_engine.common.recipe")
    return te, recipes


def make_te_recipe(recipes: Any, name: str) -> Any:
    """Build the Transformer Engine scaling recipe for ``name``."""
    if name == "fp8":
        return recipes.DelayedScaling()
    if name == "mxfp8":
        return recipes.MXFP8BlockScaling()
    if name == "nvfp4":
        return recipes.NVFP4BlockScaling()
    raise ValueError(f"unknown recipe '{name}'")


def swap_te_linears(model: TabICLv2, te: Any) -> int:
    """Swap eligible ``Linear`` modules for ``te.Linear``.

    Transformer Engine low-precision GEMMs require both feature
    dimensions to be multiples of 16 and the product of the leading
    dimensions to be divisible by 8. Ineligible layers stay in bf16:
    the feature-grouping projection and output head (feature dims), and
    the :class:`~sdm.nn.QASSMax` scaling MLPs (which can see a single
    key-length row, so their leading product can be 1).
    """
    from sdm.nn import QASSMax

    excluded: set[int] = set()
    for module in model.modules():
        if isinstance(module, QASSMax):
            excluded.update(id(m) for m in module.modules())

    swapped = 0
    for module in model.modules():
        if id(module) in excluded:
            continue
        for name, child in list(module.named_children()):
            if (
                isinstance(child, torch.nn.Linear)
                and child.in_features % 16 == 0
                and child.out_features % 16 == 0
            ):
                replacement = te.Linear(
                    child.in_features,
                    child.out_features,
                    bias=child.bias is not None,
                    params_dtype=child.weight.dtype,
                    device=child.weight.device,
                )
                with torch.no_grad():
                    replacement.weight.copy_(child.weight)
                    if child.bias is not None:
                        replacement.bias.copy_(child.bias)
                setattr(module, name, replacement)
                swapped += 1
    return swapped


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
        # Per-row worst logit shift relative to that row's own decision
        # margin: a flip-risk statistic rather than a global average.
        margin = (top2[..., 0] - top2[..., 1]).clamp(min=1e-9)
        delta = (out - ref).abs().amax(-1)
        block["top1_agreement"] = agree
        block["margin_ratio"] = (delta / margin).median().item()
        block["pass"] = agree >= 0.995 and block["margin_ratio"] < 0.1
    else:
        # Scale errors by each row's predicted quantile span (robust to
        # near-zero central quantiles and independent of quantile count).
        span = (ref.amax(-1) - ref.amin(-1)).clamp(min=1e-9)
        rel = (out - ref).abs() / span.unsqueeze(-1)
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
    precision, pin_sdpa, compile_kwargs, te_recipe = CONFIGS[config]
    device = torch.device("cuda")
    result: dict[str, Any] = dict(spec)

    model = TabICLv2(pretrained=True, device=device)
    apply_precision(model, precision)

    # Context factories: `sdpa_kernel` is a generator-based context manager
    # and therefore single-use, so build a fresh instance per call.
    context_factories: list[Any] = []
    if pin_sdpa:
        context_factories.append(
            lambda: sdpa_kernel(
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
        context_factories.append(
            lambda: torch.amp.autocast("cuda", torch.bfloat16)
        )
    if te_recipe is not None:
        te, recipes = load_transformer_engine()
        result["te_swapped_linears"] = swap_te_linears(model, te)
        fp8_recipe = make_te_recipe(recipes, te_recipe)
        context_factories.append(
            lambda: te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe)
        )

    def call(x: Tensor, y: Tensor) -> Tensor:
        with ExitStack() as stack:
            for factory in context_factories:
                stack.enter_context(factory())
            return model(x, y)

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

        # Table-stream: fresh shapes, including any recompiles. Jittered
        # shapes can violate low-precision GEMM divisibility constraints;
        # such failures are counted rather than aborting the cell.
        stream_errors = 0
        stream_start = time.perf_counter()
        for jitter in range(1, STREAM_LENGTH + 1):
            xs, ys = make_table(
                workload, task, seed=jitter, device=device, jitter=jitter
            )
            xs, ys = cast_inputs(xs, ys, precision)
            try:
                call(xs, ys)
            except (RuntimeError, ValueError):
                # Low-precision GEMMs constrain shapes (for example fp8
                # requires leading-dimension products divisible by 8), so
                # arbitrary table sizes may need padding.
                stream_errors += 1
        torch.cuda.synchronize()
        result["table_stream_per_table_s"] = (
            time.perf_counter() - stream_start
        ) / STREAM_LENGTH
        if stream_errors:
            result["table_stream_errors"] = stream_errors
            result["table_stream_per_table_s"] = None
    else:  # fit/predict route.
        num_train = WORKLOADS[workload][3]
        x_train, x_test = x[..., :num_train, :], x[..., num_train:, :]

        def fit_predict() -> Tensor:
            with ExitStack() as stack:
                for factory in context_factories:
                    stack.enter_context(factory())
                model.fit(x_train, y)
                pred = model.predict(x_test)
                model.clear()
                return pred

        def predict_only() -> Tensor:
            with ExitStack() as stack:
                for factory in context_factories:
                    stack.enter_context(factory())
                return model.predict(x_test)

        out = fit_predict().clone()
        fit_predict()  # Warmup both graphs.
        torch.cuda.synchronize()
        with ExitStack() as stack:
            for factory in context_factories:
                stack.enter_context(factory())
            start = time.perf_counter()
            model.fit(x_train, y)
            torch.cuda.synchronize()
            result["fit_s"] = time.perf_counter() - start
        predict_only()
        result.update(timed_loop(predict_only))
        model.clear()

    result["peak_mem_mb"] = torch.cuda.max_memory_allocated() / 2**20

    if route == "oneshot":
        # Pool the accuracy check over several seeded tables so gates do
        # not hinge on a handful of test rows from a single table.
        pooled = []
        for seed in range(5):
            xs, ys = make_table(workload, task, seed=seed, device=device)
            xs, ys = cast_inputs(xs, ys, precision)
            pooled.append(call(xs, ys).clone().float().cpu())
        out = torch.stack(pooled)

    # Accuracy vs the fp32 reference produced by the c0 cell.
    ref_path = os.path.join(workdir, f"ref-{workload}-{task}-{route}.pt")
    if config == "c0-fp32" and not spec.get("sentinel"):
        torch.save(out.float().cpu(), ref_path)
        result["accuracy"] = {"pass": True, "is_reference": True}
    elif config == "c0-fp32":
        result["accuracy"] = {"pass": True, "is_sentinel": True}
    elif os.path.exists(ref_path):
        ref = torch.load(ref_path, map_location="cpu")
        result["accuracy"] = accuracy_block(task, out.float().cpu(), ref)
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
    parser.add_argument(
        "--configs",
        default=",".join(CONFIGS),
        help="Comma-separated subset of configurations to benchmark "
        f"(default: all). Choices: {', '.join(CONFIGS)}.",
    )
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

    selected = [c.strip() for c in args.configs.split(",") if c.strip()]
    unknown = [c for c in selected if c not in CONFIGS]
    if unknown:
        raise SystemExit(f"unknown --configs entries: {', '.join(unknown)}")
    # The accuracy reference must exist before any dependent cell runs.
    selected = ["c0-fp32"] + [c for c in selected if c != "c0-fp32"]

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
        for config in selected
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
    # Rank by the worse of steady-state and per-fresh-table cost so a
    # config that recompiles per shape cannot win on p50 alone, and keep
    # the example recipe in the promoted set.
    top = [
        r["config"]
        for r in sorted(
            scored,
            key=lambda r: (
                max(
                    r["p50_s"],
                    r.get("table_stream_per_table_s") or r["p50_s"],
                ),
                r["p50_s"],
            ),
        )[:2]
    ]
    if "c12-autocast-compile" in selected:
        top = list(dict.fromkeys([*top, "c12-autocast-compile"]))
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
