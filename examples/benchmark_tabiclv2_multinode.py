r"""Multi-node scaling benchmark for :class:`TabICLv2` table streams.

Tables are independent, so the efficient multi-node layout is data
parallelism: every rank serves its own shard of the table stream with
the full single-GPU recipe (bucketed padding, regional compilation, and
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

The benchmark reports, per active world size (strong/weak scaling
phases): aggregate tables/second, per-rank throughput spread, scaling
efficiency against the single-GPU baseline, and startup cost
distributions (model load, compile, bucket warmup). Rank agreement is
enforced before any timing: every rank predicts the same broadcast
probe table through the padded, length-masked path the phases actually
time; per-rank deviations from rank 0 are gated on decision margins
(independently compiled ranks may autotune different kernels, so
bitwise identity is not required), the per-rank spread is stored in the
report, and a failing gate aborts every rank with a nonzero exit code -
no timing phase runs on unverified ranks.
"""

import argparse
import json
import os
import time
from typing import Any

import torch
import torch.distributed as dist
from torch import Tensor

from benchmark_tabiclv2 import (
    ROW_BUCKETS,
    make_table,
    pad_to_buckets,
    reachable_buckets,
)
from sdm import Recipe
from sdm.models import TabICLv2
from sdm.nn import (
    InducedTransformerBlock,
    TransformerBlock,
    enable_cudnn_varlen,
)


def call(model: TabICLv2, x: Tensor, y: Tensor, **kwargs: Any) -> Tensor:
    """Run the model on a concatenated context+query table.

    ``x`` holds the in-context rows followed by the query rows; ``y`` holds
    the in-context targets. A pass-through recipe keeps the ensemble
    dimension and is required by the padded cells: fitted pre-processing
    would derive its state from the padded rows, violating the
    ``seqused_*`` contract.
    """
    num_train = y.size(-1)
    out = model(
        x[..., :num_train, :],
        y.unsqueeze(-1),
        x[..., num_train:, :],
        recipe=Recipe(),
        **kwargs,
    )
    return out.numerical[0]


def init_distributed() -> tuple[int, int, torch.device]:
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
        dist.init_process_group(backend="nccl", rank=rank, world_size=world)
    return rank, world, device


def barrier(world: int, device: torch.device) -> None:
    """Synchronize all ranks (no-op for single-process runs)."""
    if world > 1:
        dist.barrier(device_ids=[device.index])


def gather_stats(value: float, world: int, device: torch.device) -> list:
    """Collect one scalar from every rank."""
    if world == 1:
        return [value]
    tensor = torch.tensor([value], device=device)
    out = [torch.zeros_like(tensor) for _ in range(world)]
    dist.all_gather(out, tensor)
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
        # Record only: the inert-path abort happens in ``main`` after a
        # cross-rank reduction, so a rank missing the optional wheel
        # cannot abort alone and strand the others at a collective.
        timings["varlen_active"] = float(enable_cudnn_varlen(True))

    start = time.perf_counter()
    for module in model.cls_model.modules():
        if isinstance(module, (TransformerBlock, InducedTransformerBlock)):
            module.compile(fullgraph=True, dynamic=True)
    model.cls_model.icl_block.head.compile(fullgraph=True, dynamic=True)
    x, y = make_table("large", "cls", seed=0, device=device)
    with torch.inference_mode():
        call(model, x.to(torch.bfloat16), y)
    torch.cuda.synchronize()
    timings["compile_s"] = time.perf_counter() - start

    start = time.perf_counter()
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
                    model,
                    xw,
                    yw,
                    seqused_train=torch.tensor(
                        train_bucket, dtype=torch.int32, device=device
                    ),
                    seqused_cols=torch.tensor(
                        64, dtype=torch.int32, device=device
                    ),
                )
    torch.cuda.synchronize()
    timings["bucket_warmup_s"] = time.perf_counter() - start
    return model, timings


def rank_agreement_gate(
    model: TabICLv2,
    rank: int,
    world: int,
    device: torch.device,
) -> tuple[bool, list[list[float]]]:
    """Gate rank agreement on the padded, masked path the phases time.

    Every rank predicts the same broadcast off-grid table through
    ``pad_to_buckets`` (so the length-masked kernels - and the
    variable-length path when enabled - are what get compared), and
    per-rank deviations from rank 0 are gated on decision margins.
    Returns the verdict and the per-rank ``[max_abs_diff, margin_ratio,
    top1]`` statistics.
    """
    x, y = make_table("large", "cls", seed=17, device=device, jitter=17)
    x = x.to(torch.bfloat16)
    if world > 1:
        dist.broadcast(x, src=0)
        dist.broadcast(y, src=0)
    x_padded, y_padded, seqused, num_test = pad_to_buckets(x, y)
    # The probe must stay off-grid: an exact-fit table returns empty
    # seqused kwargs and would silently gate the unmasked path again.
    assert seqused
    with torch.inference_mode():
        out = call(model, x_padded, y_padded, **seqused)[
            ..., :num_test, :
        ].float()
    if world == 1:
        return True, [[0.0, 0.0, 1.0]]
    reference = out.clone()
    dist.broadcast(reference, src=0)
    # Ranks compile independently, so autotuned kernel choices (and thus
    # reduction orders) may differ; gate on decision margins and record
    # the raw spread rather than requiring bitwise identity.
    top2 = reference.topk(2, dim=-1).values
    margin = (top2[..., 0] - top2[..., 1]).clamp(min=1e-9)
    margin_ratio = ((out - reference).abs().amax(-1) / margin).median()
    stats = torch.stack(
        [
            (out - reference).abs().max(),
            margin_ratio,
            (out.argmax(-1) == reference.argmax(-1)).float().mean(),
        ]
    )
    gathered = [torch.zeros_like(stats) for _ in range(world)]
    dist.all_gather(gathered, stats)
    worst_margin = max(g[1].item() for g in gathered)
    worst_top1 = min(g[2].item() for g in gathered)
    if dist.get_rank() == 0:
        print(
            f"rank agreement: max_abs_diff="
            f"{max(g[0].item() for g in gathered):.3e} "
            f"worst_margin_ratio={worst_margin:.5f} "
            f"worst_top1={worst_top1:.4f}",
            flush=True,
        )
    verdict = worst_top1 >= 0.995 and worst_margin < 0.05
    return verdict, [[v.item() for v in g] for g in gathered]


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
            jitter = seed_base + rank * 1000 + count
            x, y = make_table(
                "large", "cls", seed=jitter, device=device, jitter=jitter
            )
            x_p, y_p, seqused, num_test = pad_to_buckets(
                x.to(torch.bfloat16), y
            )
            call(model, x_p, y_p, **seqused)[..., :num_test, :]
            count += 1
    torch.cuda.synchronize()
    return count, time.perf_counter() - start


def main() -> None:
    """Run the scaling phases and write the report from rank 0."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="scaling.json")
    parser.add_argument("--duration", type=float, default=6.0)
    parser.add_argument("--use-varlen", action="store_true")
    args = parser.parse_args()

    rank, world, device = init_distributed()
    torch.set_float32_matmul_precision("high")

    model, timings = prepare_model(device, use_varlen=args.use_varlen)
    if args.use_varlen:
        active = torch.tensor(timings["varlen_active"], device=device)
        if world > 1:
            dist.all_reduce(active, op=dist.ReduceOp.MIN)
        if active.item() == 0.0:
            if rank == 0:
                print(
                    "--use-varlen requested but the cuDNN variable-length "
                    "path is inert on at least one rank "
                    "(nvidia-cudnn-frontend missing or unimportable) - "
                    "aborting instead of publishing boolean-mask numbers "
                    "under the variable-length label",
                    flush=True,
                )
            if world > 1:
                dist.destroy_process_group()
            raise SystemExit(1)
    report: dict = {
        "world_size": world,
        "row_buckets": ROW_BUCKETS,
        "startup": {
            key: gather_stats(value, world, device)
            for key, value in timings.items()
        },
    }

    agreement, agreement_stats = rank_agreement_gate(
        model=model,
        rank=rank,
        world=world,
        device=device,
    )
    report["rank_agreement"] = agreement
    report["rank_agreement_stats"] = agreement_stats
    if not agreement:
        # Abort on every rank symmetrically (the verdict is identical
        # everywhere, so no rank reaches the phase barriers) - timing an
        # unverified fleet would publish numbers the gate disowns.
        if rank == 0:
            with open(args.out, "w") as handle:
                json.dump(report, handle, indent=2)
            print(
                "rank agreement FAILED - aborting before any timing",
                flush=True,
            )
        if world > 1:
            dist.destroy_process_group()
        raise SystemExit(1)

    # Strong/weak scaling: phases with the first k ranks active. Inactive
    # ranks wait at the barriers, so every phase runs on an otherwise
    # idle machine set.
    phases = sorted({1, 4, 8, 16, 24, 32, world} & set(range(1, world + 1)))
    report["phases"] = {}
    for active in phases:
        barrier(world, device)
        if rank < active:
            count, elapsed = serve_tables(
                model, rank, args.duration, device, seed_base=10_000 * active
            )
        else:
            count, elapsed = 0, 0.0
        barrier(world, device)
        counts = gather_stats(float(count), world, device)
        elapsed_all = gather_stats(elapsed, world, device)
        active_counts = list(counts[:active])
        active_elapsed = [e for e in elapsed_all[:active] if e > 0]
        throughput = (
            sum(active_counts) / max(active_elapsed) if active_elapsed else 0.0
        )
        report["phases"][str(active)] = {
            "tables_total": sum(active_counts),
            "tables_per_s": throughput,
            "per_rank_tables": active_counts,
        }
        if rank == 0:
            print(
                f"phase ranks={active}: {throughput:.2f} tables/s "
                f"(per-rank {min(active_counts):.0f}-"
                f"{max(active_counts):.0f})",
                flush=True,
            )

    if rank == 0:
        base = report["phases"].get("1", {}).get("tables_per_s", 0.0)
        for active, phase in report["phases"].items():
            phase["efficiency_vs_1"] = (
                phase["tables_per_s"] / (int(active) * base) if base else None
            )
        with open(args.out, "w") as handle:
            json.dump(report, handle, indent=2)
        print(json.dumps(report["phases"], indent=2), flush=True)
        print(f"rank agreement: {agreement}", flush=True)

    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
