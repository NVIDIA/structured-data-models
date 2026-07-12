r"""Multi-node scaling benchmark for :class:`TabICLv2` table streams.

Tables are independent, so the efficient multi-node layout is data
parallelism: every rank serves its own shard of the table stream with
the full single-GPU recipe (bucketed padding, regional compilation, and
optionally cuDNN variable-length attention), and NCCL is used only for
coordination and metric collection - never inside the serving path.
Tensor-parallel execution of a single table was evaluated and rejected:
at 128-512 channels the per-layer collectives dominate the sub-millisecond
layer compute, and single-table latency is already launch-bound.

Launch under SLURM (one task per GPU)::

    srun -N10 --ntasks-per-node=4 --gres=gpu:4 \
        python examples/benchmark_tabiclv2_multinode.py --out scaling.json

The benchmark reports, per active world size (strong/weak scaling
phases): aggregate tables/second, per-rank throughput spread, scaling
efficiency against the single-GPU baseline, and startup cost
distributions (model load, compile, bucket warmup). Rank agreement is
gated before any timing: every rank predicts the same broadcast probe
table, and per-rank deviations from rank 0 are gated on decision
margins (independently compiled ranks may autotune different kernels,
so bitwise identity is not required; the raw spread is reported).
"""

import argparse
import json
import os
import time

import torch
import torch.distributed as dist
from sdm.models import TabICLv2
from sdm.nn import InducedTransformerBlock, TransformerBlock

from benchmark_tabiclv2 import ROW_BUCKETS, make_table, pad_to_buckets


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
        from sdm.nn import enable_cudnn_varlen

        timings["varlen_active"] = float(enable_cudnn_varlen(True))

    start = time.perf_counter()
    for module in model.cls_model.modules():
        if isinstance(module, (TransformerBlock, InducedTransformerBlock)):
            module.compile(fullgraph=True, dynamic=True)
    model.cls_model.head.compile(fullgraph=True, dynamic=True)
    x, y = make_table("large", "cls", seed=0, device=device)
    with torch.inference_mode():
        model(x.to(torch.bfloat16), y)
    torch.cuda.synchronize()
    timings["compile_s"] = time.perf_counter() - start

    start = time.perf_counter()
    with torch.inference_mode():
        for train_bucket, test_bucket in (
            (6144, 2048),
            (6144, 3072),
            (8192, 3072),
        ):
            xw = torch.zeros(
                train_bucket + test_bucket,
                72,
                device=device,
                dtype=torch.bfloat16,
            )
            yw = torch.zeros(train_bucket, dtype=torch.long, device=device)
            for _ in range(2):
                model(
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
) -> bool:
    """Every rank predicts the same broadcast table; must match rank 0."""
    x, y = make_table("large", "cls", seed=0, device=device)
    x = x.to(torch.bfloat16)
    if world > 1:
        dist.broadcast(x, src=0)
        dist.broadcast(y, src=0)
    with torch.inference_mode():
        out = model(x, y).float()
    if world == 1:
        return True
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
    return worst_top1 >= 0.995 and worst_margin < 0.05


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
            model(x_p, y_p, **seqused)[..., :num_test, :]
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
    report: dict = {
        "world_size": world,
        "row_buckets": ROW_BUCKETS,
        "startup": {
            key: gather_stats(value, world, device)
            for key, value in timings.items()
        },
    }

    agreement = rank_agreement_gate(model, rank, world, device)
    report["rank_agreement"] = agreement

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
