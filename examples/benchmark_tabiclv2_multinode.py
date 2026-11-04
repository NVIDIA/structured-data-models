# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Multi-node scaling benchmark for :class:`TabICLv2` table streams.

Tables are independent, so the efficient multi-node layout is data
parallelism: every rank serves its own shard of the table stream with
the full single-GPU recipe (bucketed padding, whole-model compilation, and
optionally cuDNN variable-length attention), and NCCL is used only for
coordination and metric collection - never inside the serving path.
Tensor-parallel execution of a single table was evaluated and rejected:
at 128-512 channels the per-layer collectives dominate the sub-millisecond
layer compute, and single-table latency is already launch-bound.
cuGraph was likewise considered and rejected: single-table in-context
inference has no graph structure to partition or traverse.

Launch under SLURM (one task per GPU). The ``env://`` rendezvous needs
``MASTER_ADDR``/``MASTER_PORT`` exported before ``srun`` (for example in
the batch script, where ``scontrol`` is available)::

    export MASTER_ADDR=$(scontrol show hostnames \
        "$SLURM_JOB_NODELIST" | head -n1)
    export MASTER_PORT=29500
    srun -N10 --ntasks-per-node=4 --gres=gpu:4 \
        python examples/benchmark_tabiclv2_multinode.py --out scaling.json

The benchmark reports, per active world size (throughput-scaling phases
- the first k ranks serve their own jittered streams for a fixed
duration, so total work grows with the rank count; there is no fixed-
total-work phase): aggregate tables/second, per-rank throughput spread,
scaling efficiency against the single-GPU baseline, and startup cost
distributions (model load, compile of the exact-fit graph family,
compile of the masked family on the first padded shape, bucket warmup
of the remaining shapes; cold-cache numbers, as every rank compiles
into a fresh Inductor cache directory). Rank
agreement is enforced before any timing: every rank predicts the same
broadcast probe table through the padded, length-masked path the phases
actually time; per-rank deviations from rank 0 are gated on decision
margins (independently compiled ranks may autotune different kernels, so
bitwise identity is not required), the per-rank spread is stored in the
report, and a failing gate aborts every rank with a nonzero exit code - no
timing phase runs on unverified ranks.
"""

import argparse
import atexit
import importlib
import json
import math
import os
import shutil
import signal
import sys
import tempfile
import time
import traceback
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import timedelta
from typing import Any

import torch
import torch.distributed as dist
from torch import Tensor
from torch.nn.attention import SDPBackend, sdpa_kernel

import sdm
from benchmark_tabiclv2 import (
    ROW_BUCKETS,
    make_table,
    pad_to_buckets,
    reachable_buckets,
    varlen_engaged,
)
from sdm import Recipe
from sdm.models import TabICLv2
from sdm.nn import cudnn_varlen_stats, enable_cudnn_varlen


def call(model: TabICLv2, x: Tensor, y: Tensor, **kwargs: Any) -> Tensor:
    """Run the model on a concatenated context+query table.

    ``x`` holds the in-context rows followed by the query rows; ``y`` holds
    the in-context targets. A pass-through recipe keeps the ensemble
    dimension and is required by the padded cells: fitted pre-processing
    would derive its state from the padded rows, violating the
    ``seqused_*`` contract.
    """
    num_train = y.size(-1)
    # Pin the SDPA backend priority as the single-node recipe does;
    # `sdpa_kernel` is a generator-based context manager and therefore
    # single-use, so build a fresh instance per call.
    with sdpa_kernel(
        [
            SDPBackend.CUDNN_ATTENTION,
            SDPBackend.FLASH_ATTENTION,
            SDPBackend.EFFICIENT_ATTENTION,
            SDPBackend.MATH,
        ],
        set_priority=True,
    ):
        out = model(
            x[..., :num_train, :],
            y.unsqueeze(-1),
            x[..., num_train:, :],
            recipe=Recipe(),
            **kwargs,
        )
    return out.numerical[0]


def init_distributed(duration_s: float) -> tuple[int, int, torch.device]:
    """Initialize NCCL from SLURM/torchrun environment variables."""
    rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0")))
    world = int(
        os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", "1"))
    )
    local_rank = int(
        os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0"))
    )
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if world > 1:
        # Inactive ranks enqueue the post-phase barrier immediately and
        # then wait ~``duration_s`` for the active ranks, so the
        # collective timeout must comfortably exceed the longest phase
        # or the NCCL watchdog aborts the job mid-benchmark.
        dist.init_process_group(
            backend="nccl",
            rank=rank,
            world_size=world,
            timeout=timedelta(seconds=max(1800.0, 4 * duration_s)),
        )
    return rank, world, device


def barrier(world: int, device: torch.device) -> None:
    """Synchronize all ranks (no-op for single-process runs)."""
    if world > 1:
        dist.barrier(device_ids=[device.index])


@contextmanager
def coordinated(
    ready: Tensor, rank: int, world: int, message: str
) -> Iterator[None]:
    """Abort all ranks if one rank's local work fails.

    Status storage is allocated before fallible work. Data collectives
    stay outside this context: transport failures cannot be recovered by
    attempting another reduction on the same process group.
    """
    ready.fill_(1)
    try:
        yield
    except Exception:  # noqa: BLE001 - reported, then a joint abort
        traceback.print_exc()
        ready.zero_()
    if world > 1:
        dist.all_reduce(ready, op=dist.ReduceOp.MIN)
    if not ready.item():
        abort(message, rank=rank, world=world)


def gather_stats(
    value: float, world: int, device: torch.device, *, ready: Tensor
) -> list[float]:
    """Collect one scalar from every rank."""
    if world == 1:
        return [float(value)]
    rank = dist.get_rank()
    with coordinated(
        ready=ready,
        rank=rank,
        world=world,
        message="statistics preparation failed on at least one rank",
    ):
        tensor = torch.tensor([float(value)], device=device)
        out = [torch.zeros_like(tensor) for _ in range(world)]
    dist.all_gather(out, tensor)
    with coordinated(
        ready=ready,
        rank=rank,
        world=world,
        message="statistics conversion failed on at least one rank",
    ):
        return [t.item() for t in out]


def prepare_model(
    device: torch.device,
    use_varlen: bool,
) -> tuple[TabICLv2, dict[str, float]]:
    """Load, compile, and bucket-warm the per-rank model (timed)."""
    timings: dict[str, float] = {}
    start = time.perf_counter()
    model = TabICLv2(pretrained=True, device=device).to(torch.bfloat16)
    timings["load_s"] = time.perf_counter() - start

    if use_varlen:
        # Record only: the engagement abort happens in ``main`` after a
        # cross-rank reduction, so a rank missing the optional wheel
        # cannot abort alone and strand the others at a collective.
        timings["varlen_active"] = float(enable_cudnn_varlen(True))

    start = time.perf_counter()
    # Whole-model compilation, matching the measured single-node serving
    # configs (regional per-block compilation is benchmarked as c17-c20
    # in benchmark_tabiclv2.py).
    model.models["classification"].compile(fullgraph=True, dynamic=True)
    x, y = make_table("large", "cls", seed=0, device=device)
    with torch.inference_mode():
        call(model, x.to(torch.bfloat16), y)
    torch.cuda.synchronize()
    timings["compile_s"] = time.perf_counter() - start

    start = time.perf_counter()
    masked_compile_s = 0.0
    with torch.inference_mode():
        # Every (train, test, column) bucket family the jittered stream
        # can reach must be warm before any timed window, or the first
        # phase absorbs fresh-shape costs and skews the scaling baseline.
        for train_bucket, test_bucket, col_bucket in sorted(
            reachable_buckets("large")
        ):
            xw = torch.zeros(
                train_bucket + test_bucket,
                col_bucket,
                device=device,
                dtype=torch.bfloat16,
            )
            yw = torch.zeros(train_bucket, dtype=torch.long, device=device)
            for _ in range(2):
                call(
                    model=model,
                    x=xw,
                    y=yw,
                    seqused_train=torch.tensor(
                        train_bucket, dtype=torch.int32, device=device
                    ),
                    seqused_cols=torch.tensor(
                        col_bucket, dtype=torch.int32, device=device
                    ),
                )
                if not masked_compile_s:
                    # The first padded call compiles the masked graph
                    # family - a second whole-model compile (~40 s cold on
                    # GB200, ~98% of this loop) - so report it as compile
                    # cost; the remaining visits are the actual per-shape
                    # warmup (~1 s in total).
                    torch.cuda.synchronize()
                    masked_compile_s = time.perf_counter() - start
    torch.cuda.synchronize()
    timings["masked_compile_s"] = masked_compile_s
    timings["bucket_warmup_s"] = time.perf_counter() - start - masked_compile_s
    if use_varlen:
        # The warmup visited every masked shape the stream can reach, so
        # the stats prove the kernel serves the phases (import
        # availability alone does not: an unsupported shape degrades to
        # the masked fallback inside the op).
        built, failed = cudnn_varlen_stats()
        timings["varlen_graphs_built"] = float(built)
        timings["varlen_build_failures"] = float(failed)
    return model, timings


def agreement_verdict(spread: Tensor) -> bool:
    """Decide rank agreement from the gathered ``[world, 3]`` statistics.

    Every statistic must be finite before the thresholds apply: a rank
    emitting NaN or inf logits is numerically invalid even when its other
    rows still agree on top-1, and Python's ``max``/``min`` would silently
    skip a NaN entry (``max([0.0, nan]) == 0.0``).
    """
    if not bool(torch.isfinite(spread).all()):
        return False
    worst_margin = spread[:, 1].max().item()
    worst_top1 = spread[:, 2].min().item()
    return worst_top1 >= 0.995 and worst_margin < 0.05


def rank_agreement_gate(
    model: TabICLv2,
    rank: int,
    world: int,
    device: torch.device,
    ready: Tensor | None = None,
) -> tuple[bool, list[list[float | None]]]:
    """Gate rank agreement on the padded, masked path the phases time.

    Every rank predicts the same broadcast off-grid table through
    ``pad_to_buckets`` (so the length-masked kernels - and the
    variable-length path when enabled - are what get compared), and
    per-rank deviations from rank 0 are gated on decision margins.
    Returns the verdict and the per-rank ``[max_abs_diff, margin_ratio,
    top1]`` statistics, with non-finite report values represented as ``None``.
    """
    # Reuse preallocated status storage so reporting a local allocation
    # failure does not require another device allocation.
    if ready is None:
        ready = torch.ones((), device=device)
    else:
        ready.fill_(1)
    # Input allocation can fail before a rank reaches the first broadcast.
    # Coordinate that failure before any peer enters the broadcast.
    x = y = None
    try:
        x, y = make_table("large", "cls", seed=17, device=device, jitter=17)
        x = x.to(torch.bfloat16)
    except Exception:  # noqa: BLE001 - reported, then a joint abort
        traceback.print_exc()
        x = y = None
        ready.zero_()
    if world > 1:
        dist.all_reduce(ready, op=dist.ReduceOp.MIN)
    if not ready.item():
        return False, []
    assert x is not None
    assert y is not None
    if world > 1:
        dist.broadcast(x, src=0)
        dist.broadcast(y, src=0)
    # The probe is per-rank work between two collectives, so a failure on
    # one rank must not leave its peers in the reference broadcast below
    # for the NCCL timeout: reduce a success flag first, like model
    # preparation and the phase loop do.
    out: Tensor | None = None
    reference: Tensor | None = None
    try:
        with torch.inference_mode():
            # Pad inside inference mode like ``serve_tables`` does: the
            # compiled graph guards the ``seqused_*`` dispatch keys, so
            # regular tensors here would compile (and then validate) a
            # third masked variant that the warmup never built and the
            # phases never run.
            x_padded, y_padded, seqused, num_test = pad_to_buckets(x, y)
            # The probe must stay off-grid: an exact-fit table returns
            # empty seqused kwargs and would silently gate the unmasked
            # path again.
            assert seqused
            out = call(model, x_padded, y_padded, **seqused)[
                ..., :num_test, :
            ].float()
        # Broadcast mutates this tensor outside inference mode.
        reference = out.clone()
    except Exception:  # noqa: BLE001 - reported, then a joint abort
        traceback.print_exc()
        ready.zero_()
    if world > 1:
        dist.all_reduce(ready, op=dist.ReduceOp.MIN)
    if not ready.item():
        if rank == 0:
            print(
                "rank agreement probe failed on at least one rank (its "
                "traceback is above)",
                flush=True,
            )
        # ``main`` aborts every rank on a failed verdict.
        return False, []
    assert out is not None
    assert reference is not None
    # A single rank is gated against itself, so non-finite logits still
    # fail the verdict below instead of being reported as agreement.
    if world > 1:
        dist.broadcast(reference, src=0)
    # Ranks compile independently, so autotuned kernel choices (and thus
    # reduction orders) may differ; gate on decision margins and record
    # the raw spread rather than requiring bitwise identity.
    stats: Tensor | None = None
    gathered: list[Tensor] = []
    try:
        top2 = reference.topk(2, dim=-1).values
        margin = (top2[..., 0] - top2[..., 1]).clamp(min=1e-9)
        margin_ratio = ((out - reference).abs().amax(-1) / margin).median()
        stats = torch.stack(
            tensors=[
                (out - reference).abs().max(),
                margin_ratio,
                (out.argmax(-1) == reference.argmax(-1)).float().mean(),
            ]
        )
        gathered = [torch.zeros_like(stats) for _ in range(world)]
    except Exception:  # noqa: BLE001 - reported, then a joint abort
        traceback.print_exc()
        ready.zero_()
    if world > 1:
        dist.all_reduce(ready, op=dist.ReduceOp.MIN)
    if not ready.item():
        return False, []
    assert stats is not None
    if world > 1:
        dist.all_gather(gathered, stats)
    else:
        gathered[0] = stats

    # Report construction is also rank-local work before later phase
    # collectives; no peer may proceed if it fails on another rank.
    verdict, reported = False, []
    try:
        spread = torch.stack(gathered)
        verdict = agreement_verdict(spread)
        reported = [
            [value if math.isfinite(value) else None for value in row]
            for row in spread.tolist()
        ]
        if rank == 0:
            # Torch reductions propagate NaN (Python's ``max`` would skip it).
            print(
                "rank agreement: "
                f"max_abs_diff={spread[:, 0].max().item():.3e} "
                f"worst_margin_ratio={spread[:, 1].max().item():.5f} "
                f"worst_top1={spread[:, 2].min().item():.4f}",
                flush=True,
            )
    except Exception:  # noqa: BLE001 - reported, then a joint abort
        traceback.print_exc()
        ready.zero_()
    if world > 1:
        dist.all_reduce(ready, op=dist.ReduceOp.MIN)
    if not ready.item():
        return False, []
    return verdict, reported


def serve_tables(
    model: TabICLv2,
    rank: int,
    duration_s: float,
    device: torch.device,
    seed_base: int,
) -> tuple[int, float]:
    """Serve bucketed jittered tables for ``duration_s``; count them."""
    count = 0
    start = time.perf_counter()
    with torch.inference_mode():
        while time.perf_counter() - start < duration_s:
            # The per-rank stride exceeds any feasible table count, so
            # rank (and, via ``seed_base``, phase) streams stay disjoint.
            jitter = seed_base + rank * 1_000_000 + count
            x, y = make_table(
                workload="large",
                task="cls",
                seed=jitter,
                device=device,
                jitter=jitter,
            )
            x_p, y_p, seqused, num_test = pad_to_buckets(
                x=x.to(torch.bfloat16), y=y
            )
            call(model, x_p, y_p, **seqused)[..., :num_test, :]
            count += 1
    torch.cuda.synchronize()
    return count, time.perf_counter() - start


def phase_summary(
    counts: list[float],
    elapsed: list[float],
    active: int,
) -> dict[str, Any]:
    """Summarize one phase from the gathered per-rank counts and windows.

    Every active rank times its own serving window, which overruns
    ``duration_s`` by up to one table latency plus the final
    synchronize, so a rank's throughput is its own ``count / elapsed``
    rather than its raw count. The aggregate divides the total by the
    longest window so it never over-reports.
    """
    active_counts = list(counts[:active])
    active_elapsed = list(elapsed[:active])
    per_rank_tables_per_s = [
        c / e if e > 0 else 0.0 for c, e in zip(active_counts, active_elapsed)
    ]
    longest = max(active_elapsed, default=0.0)
    fastest = max(per_rank_tables_per_s, default=0.0)
    return {
        "tables_total": sum(active_counts),
        "tables_per_s": sum(active_counts) / longest if longest > 0 else 0.0,
        "per_rank_tables": active_counts,
        "per_rank_elapsed_s": active_elapsed,
        "per_rank_tables_per_s": per_rank_tables_per_s,
        # Relative gap between the fastest and slowest active rank; a
        # lagging rank is otherwise invisible in the aggregate.
        "per_rank_spread": (
            (fastest - min(per_rank_tables_per_s)) / fastest
            if fastest > 0
            else 0.0
        ),
    }


def abort(message: str, rank: int, world: int) -> None:
    """Exit every rank symmetrically after rank 0 reports ``message``."""
    if rank == 0:
        print(message, flush=True)
    if world > 1:
        dist.destroy_process_group()
    raise SystemExit(1)


def main() -> None:
    """Run the scaling phases and write the report from rank 0."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--out", default="scaling.json")
    parser.add_argument("--duration", type=float, default=6.0)
    parser.add_argument("--use-varlen", action="store_true")
    args = parser.parse_args()

    rank, world, device = init_distributed(args.duration)
    # Allocate once, before any recoverable local work. A failure to
    # establish this initial control storage cannot itself be coordinated.
    ready = torch.ones((), device=device)
    model = None
    timings: dict[str, float] = {}
    with coordinated(
        ready=ready,
        rank=rank,
        world=world,
        message="model preparation failed on at least one rank",
    ):
        torch.set_float32_matmul_precision("high")
        # Fresh caches keep startup costs cold on every rank.
        cache_dir = tempfile.mkdtemp(prefix="inductor-multinode-")
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = cache_dir
        atexit.register(shutil.rmtree, cache_dir, ignore_errors=True)
        # An elastic agent's SIGTERM must also reach the cache cleanup.
        signal.signal(signal.SIGTERM, lambda *_: sys.exit(1))
        model, timings = prepare_model(device, use_varlen=args.use_varlen)
    assert model is not None
    if args.use_varlen:
        # Prepare every reduction input before any peer enters it.
        with coordinated(
            ready=ready,
            rank=rank,
            world=world,
            message="variable-length statistics failed on at least one rank",
        ):
            engaged = torch.tensor(
                data=[
                    timings["varlen_active"],
                    timings["varlen_graphs_built"],
                ],
                device=device,
            )
            failures = torch.tensor(
                data=timings["varlen_build_failures"], device=device
            )
        if world > 1:
            dist.all_reduce(engaged, op=dist.ReduceOp.MIN)
            dist.all_reduce(failures, op=dist.ReduceOp.MAX)
        with coordinated(
            ready=ready,
            rank=rank,
            world=world,
            message="variable-length verdict failed on at least one rank",
        ):
            active, built = engaged.tolist()
            varlen_ok = bool(active) and varlen_engaged(
                built=int(built), failed=int(failures.item()), exercised=True
            )
        if not varlen_ok:
            abort(
                "--use-varlen requested but the cuDNN variable-length path "
                "did not fully engage on at least one rank "
                "(nvidia-cudnn-frontend missing, or a warmed shape degraded "
                "to the masked fallback) - aborting instead of publishing "
                "boolean-mask numbers under the variable-length label",
                rank=rank,
                world=world,
            )
    report: dict[str, Any] = {}
    with coordinated(
        ready=ready,
        rank=rank,
        world=world,
        message="report setup failed on at least one rank",
    ):
        environment = {
            "device": torch.cuda.get_device_name(device),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            # cuDNN plan builds can reuse the driver cache even with a
            # fresh Inductor cache; record this distinction.
            "cuda_cache_disabled": os.environ.get("CUDA_CACHE_DISABLE") == "1",
            "cuda_cache_path": os.environ.get("CUDA_CACHE_PATH"),
            "cudnn": torch.backends.cudnn.version(),
            "sdm": sdm.__version__,
        }
        if args.use_varlen:
            environment["cudnn_frontend"] = importlib.import_module(
                name="cudnn"
            ).__version__
        report = {
            "world_size": world,
            "duration_s": args.duration,
            "use_varlen": args.use_varlen,
            "environment": environment,
            "row_buckets": ROW_BUCKETS,
            "startup": {},
        }
        startup_keys = tuple(timings)
    for key in startup_keys:
        values = gather_stats(timings[key], world, device, ready=ready)
        with coordinated(
            ready=ready,
            rank=rank,
            world=world,
            message="startup report failed on at least one rank",
        ):
            report["startup"][key] = values

    agreement, agreement_stats = rank_agreement_gate(
        model=model, rank=rank, world=world, device=device, ready=ready
    )
    with coordinated(
        ready=ready,
        rank=rank,
        world=world,
        message="agreement report failed on at least one rank",
    ):
        report["rank_agreement"] = agreement
        report["rank_agreement_stats"] = agreement_stats
        if not agreement and rank == 0:
            with open(args.out, "w") as handle:
                json.dump(report, handle, indent=2, allow_nan=False)
    if not agreement:
        abort(
            "rank agreement FAILED - aborting before any timing",
            rank=rank,
            world=world,
        )

    # First k ranks serve their independent streams for the same duration.
    with coordinated(
        ready=ready,
        rank=rank,
        world=world,
        message="phase setup failed on at least one rank",
    ):
        phases = sorted(
            {1, 4, 8, 16, 24, 32, world} & set(range(1, world + 1))
        )
        report["phases"] = {}
    for active in phases:
        barrier(world, device)
        count, elapsed = 0, 0.0
        with coordinated(
            ready=ready,
            rank=rank,
            world=world,
            message=(
                f"serving failed on at least one rank in phase ranks={active}"
            ),
        ):
            if rank < active:
                count, elapsed = serve_tables(
                    model=model,
                    rank=rank,
                    duration_s=args.duration,
                    device=device,
                    seed_base=100_000_000 * active,
                )
        barrier(world, device)
        counts = gather_stats(count, world, device, ready=ready)
        elapsed_all = gather_stats(elapsed, world, device, ready=ready)
        with coordinated(
            ready=ready,
            rank=rank,
            world=world,
            message="phase report failed on at least one rank",
        ):
            phase = phase_summary(counts, elapsed_all, active)
            report["phases"][str(active)] = phase
            if rank == 0:
                per_rank = phase["per_rank_tables_per_s"]
                row = (
                    f"phase ranks={active}: "
                    f"{phase['tables_per_s']:.2f} tables/s "
                    f"(per-rank {min(per_rank):.2f}-{max(per_rank):.2f} "
                    "tables/s, "
                    f"spread {100 * phase['per_rank_spread']:.1f}%)"
                )
                if phase["per_rank_spread"] > 0.05:
                    row += " SPREAD"
                print(row, flush=True)

    with coordinated(
        ready=ready,
        rank=rank,
        world=world,
        message="final report failed on at least one rank",
    ):
        if rank == 0:
            base = report["phases"].get("1", {}).get("tables_per_s", 0.0)
            for active, phase in report["phases"].items():
                phase["efficiency_vs_1"] = (
                    phase["tables_per_s"] / (int(active) * base)
                    if base
                    else None
                )
            with open(args.out, "w") as handle:
                json.dump(report, handle, indent=2, allow_nan=False)
            print(
                json.dumps(report["phases"], indent=2, allow_nan=False),
                flush=True,
            )
            print(f"rank agreement: {agreement}", flush=True)

    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
