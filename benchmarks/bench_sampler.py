r"""Benchmark relational neighbor sampling: SDM in-memory vs. DiskGraph.

Compares the two sampling backends on an identical synthetic star schema
(``users`` and ``merchants`` linked by ``transactions``) with identical
seeds, fanouts, and (optionally) temporal constraints.

Workflow (each phase is a separate process so peak RSS is meaningful):

    # 1. Generate a dataset (scale via flags):
    python benchmarks/bench_sampler.py gen --root /tmp/bench --num-users 100_000 \
        --num-merchants 10_000 --num-transactions 5_000_000

    # 2. Benchmark the SDM sampler (FK joins in memory + pyg_lib):
    python benchmarks/bench_sampler.py sdm --root /tmp/bench

    # 3. Build the DiskGraph index (one-off; reported separately):
    python benchmarks/bench_sampler.py ingest --root /tmp/bench \
        --diskgraph-repo ~/work/kumo-diskgraph

    # 4. Benchmark the DiskGraph sampler (requires the `diskgraph` wheel):
    python benchmarks/bench_sampler.py diskgraph --root /tmp/bench

    # 5. End-to-end SDM run (sampling + KumoRFM forward) to measure what
    #    fraction of wall time sampling accounts for:
    python benchmarks/bench_sampler.py e2e --root /tmp/bench

Fairness notes:
- Both phases derive seed node ids from the same RNG seed, so they sample
  from identical seed batches. DiskGraph seeds are dense node ids; since
  ``user_id`` is generated as ``0..num_users-1`` and ingested in order,
  dense id == primary key == row index.
- For cold-cache DiskGraph numbers, drop the page cache between runs:
  ``sync && echo 3 | sudo tee /proc/sys/vm/drop_caches``.
- SDM setup time (FK joins + CSC build) is the analogue of DiskGraph
  ingest time; both are reported separately from per-batch latency.
"""

import argparse
import json
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

EPOCH_START_US = 1_577_836_800_000_000  # 2020-01-01
EPOCH_SPAN_US = 2 * 365 * 24 * 3600 * 1_000_000  # ~2 years


def _generate(args: argparse.Namespace) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    root = Path(args.root)
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    users = pa.table(
        {
            "user_id": np.arange(args.num_users, dtype=np.int64),
            "age": rng.uniform(18, 90, args.num_users),
        }
    )
    pq.write_table(users, data_dir / "users.parquet")

    merchants = pa.table(
        {
            "merchant_id": np.arange(args.num_merchants, dtype=np.int64),
            "score": rng.uniform(0, 5, args.num_merchants),
        }
    )
    pq.write_table(merchants, data_dir / "merchants.parquet")

    timestamps = rng.integers(
        EPOCH_START_US,
        EPOCH_START_US + EPOCH_SPAN_US,
        args.num_transactions,
        dtype=np.int64,
    )
    transactions = pa.table(
        {
            "txn_id": np.arange(args.num_transactions, dtype=np.int64),
            "user_id": rng.integers(
                0, args.num_users, args.num_transactions, dtype=np.int64
            ),
            "merchant_id": rng.integers(
                0, args.num_merchants, args.num_transactions, dtype=np.int64
            ),
            "timestamp": timestamps,
            "amount": rng.exponential(30.0, args.num_transactions),
        }
    )
    pq.write_table(transactions, data_dir / "transactions.parquet")

    config = {
        "dataset_name": "sdm_bench",
        "num_shards": 1,
        "tables": [
            {
                "name": "users",
                "columns": [
                    {"name": "user_id", "ingested_dtype": "int64"},
                    {"name": "age", "ingested_dtype": "float64"},
                ],
                "primary_key": "user_id",
                "source": {
                    "type": "parquet",
                    "paths": [str(data_dir / "users.parquet")],
                },
            },
            {
                "name": "merchants",
                "columns": [
                    {"name": "merchant_id", "ingested_dtype": "int64"},
                    {"name": "score", "ingested_dtype": "float64"},
                ],
                "primary_key": "merchant_id",
                "source": {
                    "type": "parquet",
                    "paths": [str(data_dir / "merchants.parquet")],
                },
            },
            {
                "name": "transactions",
                "columns": [
                    {"name": "txn_id", "ingested_dtype": "int64"},
                    {"name": "user_id", "ingested_dtype": "int64"},
                    {"name": "merchant_id", "ingested_dtype": "int64"},
                    {"name": "timestamp", "ingested_dtype": "timestamp"},
                    {"name": "amount", "ingested_dtype": "float64"},
                ],
                "primary_key": "txn_id",
                "create_time": "timestamp",
                "fkeys": [
                    {
                        "src_column": "user_id",
                        "dst_table": "users",
                        "reverse": {},
                    },
                    {
                        "src_column": "merchant_id",
                        "dst_table": "merchants",
                        "reverse": {},
                    },
                ],
                "source": {
                    "type": "parquet",
                    "paths": [str(data_dir / "transactions.parquet")],
                },
            },
        ],
    }
    config_path = root / "diskgraph_config.json"
    config_path.write_text(json.dumps(config, indent=2))

    meta = {
        "num_users": args.num_users,
        "num_merchants": args.num_merchants,
        "num_transactions": args.num_transactions,
        "seed": args.seed,
    }
    (root / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Wrote dataset to {data_dir} and config to {config_path}")


def _seed_batches(
    args: argparse.Namespace, num_users: int
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Identical (seed_ids, seed_timestamps) batches for both backends."""
    rng = np.random.default_rng(args.seed + 1)
    total = args.num_batches + args.warmup
    seeds = [
        rng.integers(0, num_users, args.batch_size, dtype=np.int64)
        for _ in range(total)
    ]
    times = [
        rng.integers(
            EPOCH_START_US,
            EPOCH_START_US + EPOCH_SPAN_US,
            args.batch_size,
            dtype=np.int64,
        )
        for _ in range(total)
    ]
    return seeds, times


def _emit(result: dict[str, Any], args: argparse.Namespace) -> None:
    """Print the result and optionally append it to a JSONL results file."""
    meta = json.loads((Path(args.root) / "meta.json").read_text())
    result = {
        **result,
        "dataset": {
            "num_users": meta["num_users"],
            "num_merchants": meta["num_merchants"],
            "num_transactions": meta["num_transactions"],
        },
        "fanout": args.fanout,
        "temporal": args.temporal,
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        // 1024,
    }
    print(json.dumps(result, indent=2))
    if args.out is not None:
        with open(args.out, "a") as f:
            f.write(json.dumps(result) + "\n")


def _report(
    backend: str,
    setup_s: float,
    latencies_s: list[float],
    args: argparse.Namespace,
    sampled: dict[str, Any],
) -> None:
    lat_ms = sorted(latency * 1000 for latency in latencies_s)
    quantiles = statistics.quantiles(lat_ms, n=100)
    result = {
        "backend": backend,
        "setup_s": round(setup_s, 3),
        "batches": len(lat_ms),
        "batch_size": args.batch_size,
        "latency_ms": {
            "p50": round(quantiles[49], 2),
            "p95": round(quantiles[94], 2),
            "p99": round(quantiles[98], 2),
            "mean": round(statistics.mean(lat_ms), 2),
        },
        "seeds_per_s": round(args.batch_size / statistics.mean(latencies_s)),
        "sampled_per_batch": sampled,
    }
    _emit(result, args)


def _load_relational_data(root: Path) -> tuple[dict[str, Any], Any]:
    import pandas as pd

    from sdm import RelationalData, TableTensor

    meta = json.loads((root / "meta.json").read_text())
    data_dir = root / "data"

    stypes: dict[str, dict[str, str]] = {
        "users": {"user_id": "id", "age": "numerical"},
        "merchants": {"merchant_id": "id", "score": "numerical"},
        "transactions": {
            "txn_id": "id",
            "user_id": "id",
            "merchant_id": "id",
            "timestamp": "datetime",
            "amount": "numerical",
        },
    }
    tables = {}
    for name in ["users", "merchants", "transactions"]:
        df = pd.read_parquet(data_dir / f"{name}.parquet")
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="us")
        tables[name] = TableTensor.from_pandas(df=df, stypes=stypes[name])

    data = RelationalData(
        tables=tables,
        relationships=[
            {
                "left_table": "transactions",
                "left_column": "user_id",
                "right_table": "users",
                "right_column": "user_id",
            },
            {
                "left_table": "transactions",
                "left_column": "merchant_id",
                "right_table": "merchants",
                "right_column": "merchant_id",
            },
        ],
    )
    return meta, data


def _make_task_table(
    seed_ids: np.ndarray,
    timestamps: np.ndarray,
    target: np.ndarray | None = None,
) -> Any:
    import pandas as pd

    from sdm import TableTensor

    columns: dict[str, Any] = {
        "user_id": seed_ids,
        "timestamp": pd.to_datetime(timestamps, unit="us"),
    }
    stypes = {"user_id": "id", "timestamp": "datetime"}
    if target is not None:
        columns["target"] = target
        stypes["target"] = "numerical"
    return TableTensor.from_pandas(
        df=pd.DataFrame(columns),
        stypes=stypes,
    )


def _bench_sdm(args: argparse.Namespace) -> None:
    import torch

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    root = Path(args.root)
    meta, data = _load_relational_data(root)

    start = time.perf_counter()
    sampler = data.sampler(
        time_columns={"transactions": "timestamp"} if args.temporal else None,
    )
    setup_s = time.perf_counter() - start

    seeds, times = _seed_batches(args, meta["num_users"])
    task_link = {
        "task_column": "user_id",
        "table": "users",
        "table_column": "user_id",
    }
    kwargs: dict[str, Any] = {
        "task_link": task_link,
        "num_neighbors": args.fanout,
    }
    if args.temporal:
        kwargs["task_time_column"] = "timestamp"

    task_tables = [
        _make_task_table(seeds[index], times[index])
        for index in range(len(seeds))
    ]

    latencies: list[float] = []
    sampled: dict[str, Any] = {}
    with torch.inference_mode():
        for index, task_table in enumerate(task_tables):
            start = time.perf_counter()
            output = sampler(task_table, **kwargs)
            elapsed = time.perf_counter() - start
            if index >= args.warmup:
                latencies.append(elapsed)
            if index == args.warmup:
                sampled = {
                    name: len(table)
                    for name, table in output.related_tables.tables.items()
                }

    _report("sdm", setup_s, latencies, args, sampled)


def _ingest(args: argparse.Namespace) -> None:
    root = Path(args.root)
    index_dir = root / "index"
    start = time.perf_counter()
    subprocess.run(
        [
            "cargo",
            "run",
            "--release",
            "--bin",
            "ingest",
            "--",
            "--source",
            str(root / "diskgraph_config.json"),
            "--dest",
            "local",
            "--output",
            str(index_dir),
        ],
        cwd=Path(args.diskgraph_repo).expanduser(),
        check=True,
    )
    print(
        json.dumps(
            {
                "backend": "diskgraph",
                "ingest_s": round(time.perf_counter() - start, 3),
                "index_dir": str(index_dir),
            }
        )
    )


def _bench_diskgraph(args: argparse.Namespace) -> None:
    try:
        import diskgraph
    except ImportError:
        sys.exit(
            "The 'diskgraph' wheel is not installed. Build it with maturin "
            "from the diskgraph-pybind crate and retry."
        )

    root = Path(args.root)
    meta = json.loads((root / "meta.json").read_text())
    index_dir = root / "index"
    if not index_dir.exists():
        sys.exit(f"No index at {index_dir}. Run the 'ingest' phase first.")

    start = time.perf_counter()
    engine = diskgraph.SamplerEngine(str(index_dir))
    setup_s = time.perf_counter() - start

    # One layer per hop; every relation (both directions) gets the same
    # fanout, matching SDM's hetero sampling over all edge types per hop.
    relations = [
        "transactions_user_id_to_users_user_id",
        "transactions_user_id_to_users_user_id_reverse",
        "transactions_merchant_id_to_merchants_merchant_id",
        "transactions_merchant_id_to_merchants_merchant_id_reverse",
    ]
    plan = {
        "layers": [
            {
                "fanouts": {
                    relation: {
                        "fanout": fanout,
                        "mode": "uniform",
                        "absent_key_policy": "uniform_fallback",
                    }
                    for relation in relations
                },
                "default_fanout": None,
            }
            for fanout in args.fanout
        ]
    }
    plan_json = json.dumps(plan)

    seeds, times = _seed_batches(args, meta["num_users"])

    latencies: list[float] = []
    sampled: dict[str, Any] = {}
    for index in range(len(seeds)):
        kwargs: dict[str, Any] = {"fused": not args.no_fused}
        if args.temporal:
            kwargs["seed_timestamps"] = {"users": times[index].tolist()}
        seed_ids = seeds[index].tolist()

        start = time.perf_counter()
        stats: dict[str, Any] = {}
        try:
            iterator = engine.sample({"users": seed_ids}, plan_json, **kwargs)
            for event in iterator:
                if event["type"] == "stats":
                    stats = json.loads(event["stats"])
        except Exception:
            print("Sampling failed. Available types:", file=sys.stderr)
            print(engine.get_type_mapping(), file=sys.stderr)
            raise
        elapsed = time.perf_counter() - start

        if index >= args.warmup:
            latencies.append(elapsed)
        if index == args.warmup:
            sampled = {
                "total_nodes": stats.get("total_nodes"),
                "total_edges": stats.get("total_edges"),
            }

    _report("diskgraph", setup_s, latencies, args, sampled)


def _bench_e2e(args: argparse.Namespace) -> None:
    import torch

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from sdm.models import KumoRFM

    root = Path(args.root)
    meta, data = _load_relational_data(root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sampler = data.sampler(
        time_columns={"transactions": "timestamp"} if args.temporal else None,
    )
    kwargs: dict[str, Any] = {
        "task_link": {
            "task_column": "user_id",
            "table": "users",
            "table_column": "user_id",
        },
        "num_neighbors": args.fanout,
    }
    if args.temporal:
        kwargs["task_time_column"] = "timestamp"

    # Random-init weights time identically to pretrained ones; --pretrained
    # downloads the real checkpoint from Hugging Face.
    model = KumoRFM(pretrained=args.pretrained, device=device)

    rng = np.random.default_rng(args.seed + 2)
    context_ids = rng.integers(
        0, meta["num_users"], args.context_size, dtype=np.int64
    )
    context_times = rng.integers(
        EPOCH_START_US,
        EPOCH_START_US + EPOCH_SPAN_US,
        args.context_size,
        dtype=np.int64,
    )
    context_target = rng.uniform(0, 100, args.context_size)
    context = _make_task_table(context_ids, context_times, context_target)

    start = time.perf_counter()
    context, related_tables = sampler(context, **kwargs).to(device)
    model.fit(
        x=context.drop_columns("target"),
        y=context["target"],
        related_tables=related_tables,
    )
    fit_s = time.perf_counter() - start

    seeds, times = _seed_batches(args, meta["num_users"])
    task_tables = [
        _make_task_table(seeds[index], times[index])
        for index in range(len(seeds))
    ]

    sample_s: list[float] = []
    predict_s: list[float] = []
    for index, task_table in enumerate(task_tables):
        start = time.perf_counter()
        output = sampler(task_table, **kwargs).to(device)
        sampled = time.perf_counter()
        model.predict(*output)
        done = time.perf_counter()
        if index >= args.warmup:
            sample_s.append(sampled - start)
            predict_s.append(done - sampled)
    model.clear()

    total_s = [s + p for s, p in zip(sample_s, predict_s)]
    result = {
        "backend": "sdm-e2e",
        "device": str(device),
        "pretrained": args.pretrained,
        "fit_s": round(fit_s, 3),
        "batches": len(total_s),
        "batch_size": args.batch_size,
        "sample_ms_p50": round(statistics.median(sample_s) * 1000, 2),
        "predict_ms_p50": round(statistics.median(predict_s) * 1000, 2),
        "sampling_fraction": round(
            sum(sample_s) / sum(total_s),
            3,
        ),
    }
    _emit(result, args)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="phase", required=True)

    gen = subparsers.add_parser("gen", help="Generate the synthetic dataset")
    gen.add_argument("--root", type=str, required=True)
    gen.add_argument("--num-users", type=int, default=100_000)
    gen.add_argument("--num-merchants", type=int, default=10_000)
    gen.add_argument("--num-transactions", type=int, default=5_000_000)
    gen.add_argument("--seed", type=int, default=42)

    ingest = subparsers.add_parser("ingest", help="Build the DiskGraph index")
    ingest.add_argument("--root", type=str, required=True)
    ingest.add_argument("--diskgraph-repo", type=str, required=True)

    for name in ["sdm", "diskgraph", "e2e"]:
        bench = subparsers.add_parser(name, help=f"Benchmark the {name} side")
        bench.add_argument("--root", type=str, required=True)
        bench.add_argument("--seed", type=int, default=42)
        bench.add_argument("--batch-size", type=int, default=1000)
        bench.add_argument("--num-batches", type=int, default=20)
        bench.add_argument("--warmup", type=int, default=3)
        bench.add_argument("--fanout", type=int, nargs="+", default=[16, 16])
        bench.add_argument("--temporal", action="store_true")
        bench.add_argument("--out", type=str, default=None)
        if name == "diskgraph":
            bench.add_argument("--no-fused", action="store_true")
        if name == "e2e":
            bench.add_argument("--context-size", type=int, default=1000)
            bench.add_argument("--pretrained", action="store_true")

    args = parser.parse_args()
    if args.phase == "gen":
        _generate(args)
    elif args.phase == "ingest":
        _ingest(args)
    elif args.phase == "sdm":
        _bench_sdm(args)
    elif args.phase == "diskgraph":
        _bench_diskgraph(args)
    elif args.phase == "e2e":
        _bench_e2e(args)


if __name__ == "__main__":
    main()
