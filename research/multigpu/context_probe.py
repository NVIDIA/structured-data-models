"""Compare native SDPA, single-rank LSE, and distributed cached ICL replay.

Run with torchrun --standalone --nproc-per-node=4 this_file.py.
Random weights test numerical behavior and kernel scaling, not model quality.
"""

import argparse
import os
import statistics
import time
from contextlib import nullcontext

import torch
import torch.distributed as dist

from sdm.cache import Cache
from sdm.models.kumo.tabular.icl import ICLBlock as TabularICL
from sdm.models.tabiclv2.icl import ICLBlock as RelationalICL
from sdm.nn.context_parallel import context_parallel


def main() -> None:
    """Measure synchronized replay and fit on the requested topology."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--family", choices=["tabular", "relational"], default="tabular"
    )
    parser.add_argument("--context", type=int, default=1024)
    parser.add_argument("--queries", type=int, default=128)
    parser.add_argument("--channels", type=int, default=512)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument(
        "--dtype", choices=["float32", "bfloat16"], default="bfloat16"
    )
    args = parser.parse_args()
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl")
    single = dist.new_group([0])
    dtype = getattr(torch, args.dtype)
    torch.manual_seed(120)
    kwargs = {
        "num_classes": 3,
        "out_channels": 3,
        "channels": args.channels,
        "num_layers": args.layers,
        "num_heads": args.heads,
        "device": f"cuda:{rank}",
    }
    model = (
        TabularICL(**kwargs, num_key_value_heads_for_query=args.kv_heads)
        if args.family == "tabular"
        else RelationalICL(**kwargs, norm_bias=True)
    ).eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.uniform_(-0.1, 0.1)
    # Generate on CPU: identical independent of rank GPU RNG offsets.
    x = torch.randn(1, args.context, args.channels).cuda()
    query = torch.randn(1, args.queries, args.channels).cuda()
    y = torch.randint(3, (1, args.context)).cuda()
    results = []
    reference = None
    try:
        with (
            torch.inference_mode(),
            torch.autocast(
                "cuda", dtype=dtype, enabled=dtype != torch.float32
            ),
        ):
            for mode in ("native_sdpa", "single_rank_lse", "context_parallel"):
                dist.barrier()
                if mode != "context_parallel" and rank != 0:
                    continue
                group = (
                    dist.group.WORLD if mode == "context_parallel" else single
                )
                scope = (
                    nullcontext()
                    if mode == "native_sdpa"
                    else context_parallel(group)
                )
                with scope:
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    cache = Cache()
                    model(x.clone(), y, cache=cache)
                    cache.freeze()
                    torch.cuda.synchronize()
                    fit_seconds = time.perf_counter() - started
                    fit_peak = torch.cuda.max_memory_allocated()
                    for _ in range(2):
                        model(query.clone(), y[..., :0], cache=cache)
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    timings = []
                    for _ in range(args.repeats):
                        torch.cuda.synchronize()
                        started = time.perf_counter()
                        prediction = model(
                            query.clone(), y[..., :0], cache=cache
                        )
                        torch.cuda.synchronize()
                        timings.append(time.perf_counter() - started)
                    difference = None
                    if rank == 0:
                        if reference is None:
                            reference = prediction.clone()
                        difference = (
                            (prediction - reference).abs().max().item()
                        )
                        torch.testing.assert_close(
                            prediction,
                            reference,
                            atol=0.005 if dtype == torch.bfloat16 else 3e-5,
                            rtol=0.02 if dtype == torch.bfloat16 else 3e-4,
                        )
                    prediction_peak = torch.cuda.max_memory_allocated()
                    record = {
                        "mode": mode,
                        "rank": rank,
                        "world": dist.get_world_size(group)
                        if mode != "native_sdpa"
                        else 1,
                        "fit_seconds": fit_seconds,
                        "fit_peak_bytes": fit_peak,
                        "cache_bytes": cache.size(),
                        "prediction_peak_bytes": prediction_peak,
                        "prediction_seconds": timings,
                        "median_seconds": statistics.median(timings),
                        "max_abs_difference": difference,
                    }
                    if mode == "context_parallel":
                        gathered = [None] * dist.get_world_size()
                        dist.all_gather_object(gathered, record)
                        if rank == 0:
                            results.extend(gathered)
                    else:
                        results.append(record)
                    del cache, prediction
            if rank == 0:
                pass
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
