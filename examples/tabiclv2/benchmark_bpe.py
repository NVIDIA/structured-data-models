"""Benchmark the current SentenceTransformer CPU path for BPE models.

Run the embedding-path benchmark with no arguments. Pass ``--tabicl`` to
also measure the optimistic impact of eliminating its non-forward work from
the full TabICLv2 fit-and-predict path.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch
from huggingface_hub import hf_hub_download

import sdm
import sdm.processing as sp
from sdm import Stype, TableTensor

DATASET_REPO = "inria-soda/STRABLE-benchmark"
DATASETS = (
    "chocolate-bar-ratings",
    "mercari",
    "covid-clinical-trials",
    "clear-corpus",
    "financial-product-complaint",
)
MODELS = (
    "sentence-transformers/all-distilroberta-v1",
    "nomic-ai/modernbert-embed-base",
    "Qwen/Qwen3-Embedding-0.6B",
)

MAX_ROWS = 2048
MAX_SEQ_LENGTH = 512
BATCH_SIZE = 32
WARMUP_RUNS = 1
RUNS = 3
SEED = 42

OUTPUT = Path("bpe_validation.csv")
TABICL_OUTPUT = Path("bpe_validation_tabicl.csv")

EMBEDDING_FIELDS = (
    "model",
    "dataset",
    "run",
    "num_rows",
    "num_text_columns",
    "num_strings",
    "num_non_ascii_strings",
    "mean_chars",
    "max_seq_length",
    "processor_s",
    "preprocess_cpu_work_s",
    "forward_gpu_s",
    "non_forward_headroom_s",
    "non_forward_headroom_share",
    "max_speedup_bound",
)
TABICL_FIELDS = (
    "model",
    "dataset",
    "run",
    "num_rows",
    "fit_s",
    "predict_s",
    "total_s",
    "embedding_headroom_bound_s",
    "end_to_end_headroom_share",
    "max_speedup_bound",
)


@dataclass(frozen=True)
class Dataset:
    name: str
    table: pa.Table
    stypes: dict[str, Stype]
    text_columns: tuple[str, ...]
    target_name: str
    num_non_ascii_strings: int
    mean_chars: float


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tabicl",
        action="store_true",
        help="Also benchmark the full TabICLv2 fit-and-predict path.",
    )
    return parser.parse_args()


def _load_dataset(name: str) -> Dataset:
    data_path = hf_hub_download(
        repo_id=DATASET_REPO,
        filename=f"{name}/data.parquet",
        repo_type="dataset",
    )
    table = pq.read_table(data_path)
    stypes = {
        column: Stype(stype)
        for column, stype in sdm.infer_stypes(
            table,
            text="infer",
        ).items()
    }
    text_columns = tuple(
        column for column, stype in stypes.items() if stype == Stype.text
    )
    if len(table) > MAX_ROWS:
        indices = np.linspace(
            0,
            len(table) - 1,
            num=MAX_ROWS,
            dtype=np.int64,
        )
        table = table.take(pa.array(indices))

    config_path = hf_hub_download(
        repo_id=DATASET_REPO,
        filename=f"{name}/config.json",
        repo_type="dataset",
    )
    with Path(config_path).open() as file:
        target_name = cast(str, json.load(file)["target_name"])

    num_non_ascii_strings = 0
    num_non_null_strings = 0
    num_chars = 0
    for column in text_columns:
        strings = pc.drop_null(table.column(column).combine_chunks())
        char_lengths = pc.utf8_length(strings)
        byte_lengths = pc.binary_length(strings)
        num_non_null_strings += len(strings)
        num_chars += pc.sum(char_lengths).as_py() or 0
        num_non_ascii_strings += (
            pc.sum(pc.greater(byte_lengths, char_lengths)).as_py() or 0
        )

    return Dataset(
        name=name,
        table=table,
        stypes=stypes,
        text_columns=text_columns,
        target_name=target_name,
        num_non_ascii_strings=num_non_ascii_strings,
        mean_chars=num_chars / num_non_null_strings,
    )


def _sync() -> None:
    torch.cuda.synchronize()


def _time(function: Any) -> tuple[Any, float]:
    _sync()
    start = time.perf_counter()
    out = function()
    _sync()
    return out, time.perf_counter() - start


@contextmanager
def _profile_model(model: Any) -> Iterator[dict[str, Any]]:
    """Time existing SentenceTransformer methods without replacing behavior."""
    profile: dict[str, Any] = {}
    original_preprocess = model.preprocess
    original_forward = model.forward

    def preprocess(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        out = original_preprocess(*args, **kwargs)
        profile["preprocess_s"] += time.perf_counter() - start
        return out

    def forward(*args: Any, **kwargs: Any) -> Any:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        out = original_forward(*args, **kwargs)
        end_event.record()
        profile["forward_events"].append((start_event, end_event))
        return out

    model.preprocess = preprocess
    model.forward = forward
    try:
        yield profile
    finally:
        model.preprocess = original_preprocess
        model.forward = original_forward


def _transform(
    processor: sp.SentenceTransformer,
    table: TableTensor,
) -> TableTensor:
    with torch.amp.autocast("cuda", torch.float16):
        return processor.transform(table)


def _benchmark_processor(
    processor: sp.SentenceTransformer,
    model_name: str,
    dataset: Dataset,
    *,
    device: torch.device,
) -> list[dict[str, object]]:
    table = TableTensor.from_arrow(
        table=dataset.table.select(dataset.text_columns),
        stypes=dict.fromkeys(dataset.text_columns, Stype.text),
        device=device,
    )
    model = processor._model.module
    tokenizer_model = type(model.tokenizer.backend_tokenizer.model).__name__
    if tokenizer_model != "BPE":
        raise TypeError(
            f"Expected a BPE tokenizer for {model_name!r}, got "
            f"{tokenizer_model!r}"
        )

    for _ in range(WARMUP_RUNS):
        out = _transform(processor, table)
        _sync()
        del out

    rows: list[dict[str, object]] = []
    with _profile_model(model) as profile:
        for run in range(RUNS):
            profile.clear()
            profile.update(preprocess_s=0.0, forward_events=[])
            out, processor_s = _time(lambda: _transform(processor, table))
            del out

            forward_gpu_s = (
                sum(
                    start.elapsed_time(end)
                    for start, end in profile["forward_events"]
                )
                / 1000
            )
            headroom_s = max(processor_s - forward_gpu_s, 0.0)
            rows.append(
                {
                    "model": model_name,
                    "dataset": dataset.name,
                    "run": run,
                    "num_rows": len(dataset.table),
                    "num_text_columns": len(dataset.text_columns),
                    "num_strings": len(dataset.table)
                    * len(dataset.text_columns),
                    "num_non_ascii_strings": (dataset.num_non_ascii_strings),
                    "mean_chars": dataset.mean_chars,
                    "max_seq_length": model.max_seq_length,
                    "processor_s": processor_s,
                    "preprocess_cpu_work_s": profile["preprocess_s"],
                    "forward_gpu_s": forward_gpu_s,
                    "non_forward_headroom_s": headroom_s,
                    "non_forward_headroom_share": headroom_s / processor_s,
                    "max_speedup_bound": (
                        processor_s / forward_gpu_s
                        if forward_gpu_s > 0
                        else math.inf
                    ),
                }
            )
    return rows


def _benchmark_tabicl(
    processor: sp.SentenceTransformer,
    model_name: str,
    tabicl: sdm.models.TabICLv2,
    dataset: Dataset,
    embedding_headroom_s: float,
    *,
    device: torch.device,
) -> list[dict[str, object]]:
    table = TableTensor.from_arrow(
        table=dataset.table,
        stypes=dataset.stypes,
        device=device,
    )
    rows: list[dict[str, object]] = []
    for run in range(RUNS):
        generator = torch.Generator(device=device).manual_seed(SEED)
        perm = torch.randperm(len(table), generator=generator, device=device)
        context_size = int(0.8 * len(table))
        context = table[perm[:context_size]]
        query = table[perm[context_size:]]
        recipe = tabicl.default_recipe().prepend_features(
            sp.StypeDispatch(text=[processor, sp.PCA(64)])
        )

        _sync()
        start = time.perf_counter()
        with torch.amp.autocast("cuda", torch.float16):
            tabicl.fit(
                x=context.drop_columns(dataset.target_name),
                y=context[:, dataset.target_name],
                recipe=recipe,
                generator=generator,
            )
        _sync()
        fit_s = time.perf_counter() - start

        start = time.perf_counter()
        with torch.amp.autocast("cuda", torch.float16):
            tabicl.predict(query.drop_columns(dataset.target_name))
        _sync()
        predict_s = time.perf_counter() - start
        total_s = fit_s + predict_s
        removable_s = min(embedding_headroom_s, total_s)
        rows.append(
            {
                "model": model_name,
                "dataset": dataset.name,
                "run": run,
                "num_rows": len(dataset.table),
                "fit_s": fit_s,
                "predict_s": predict_s,
                "total_s": total_s,
                "embedding_headroom_bound_s": embedding_headroom_s,
                "end_to_end_headroom_share": removable_s / total_s,
                "max_speedup_bound": (
                    total_s / (total_s - removable_s)
                    if removable_s < total_s
                    else math.inf
                ),
            }
        )
    return rows


def _write_csv(
    path: Path,
    fields: Sequence[str],
    rows: Sequence[dict[str, object]],
) -> None:
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _print_summary(
    embedding_rows: Sequence[dict[str, object]],
    tabicl_rows: Sequence[dict[str, object]],
) -> None:
    print("\nEmbedding-path medians")
    print("model | processor (s) | non-forward bound | max speedup bound")
    for model_name in MODELS:
        rows = [row for row in embedding_rows if row["model"] == model_name]
        processor_s = statistics.median(
            cast(float, row["processor_s"]) for row in rows
        )
        headroom = statistics.median(
            cast(float, row["non_forward_headroom_share"]) for row in rows
        )
        speedup = statistics.median(
            cast(float, row["max_speedup_bound"]) for row in rows
        )
        print(
            f"{model_name} | {processor_s:.3f} | {headroom:.1%} | "
            f"{speedup:.2f}x"
        )

    if not tabicl_rows:
        return

    print("\nTabICLv2 medians")
    print("model | total (s) | end-to-end bound | max speedup bound")
    for model_name in MODELS:
        rows = [row for row in tabicl_rows if row["model"] == model_name]
        total_s = statistics.median(
            cast(float, row["total_s"]) for row in rows
        )
        headroom = statistics.median(
            cast(float, row["end_to_end_headroom_share"]) for row in rows
        )
        speedup = statistics.median(
            cast(float, row["max_speedup_bound"]) for row in rows
        )
        print(
            f"{model_name} | {total_s:.3f} | {headroom:.1%} | {speedup:.2f}x"
        )


def main() -> None:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA")

    device = torch.device("cuda")
    datasets = [_load_dataset(name) for name in DATASETS]
    tabicl = sdm.models.TabICLv2(device=device) if args.tabicl else None
    embedding_rows: list[dict[str, object]] = []
    tabicl_rows: list[dict[str, object]] = []

    for model_name in MODELS:
        print(f"\nLoading {model_name}")
        processor = sp.SentenceTransformer(
            model_name,
            batch_size=BATCH_SIZE,
        ).to(device)
        model = processor._model.module
        if model.max_seq_length is None:
            model.max_seq_length = MAX_SEQ_LENGTH
        else:
            model.max_seq_length = min(
                model.max_seq_length,
                MAX_SEQ_LENGTH,
            )

        for dataset in datasets:
            print(f"Benchmarking {dataset.name}")
            rows = _benchmark_processor(
                processor,
                model_name,
                dataset,
                device=device,
            )
            embedding_rows.extend(rows)
            _write_csv(OUTPUT, EMBEDDING_FIELDS, embedding_rows)

            if tabicl is not None:
                embedding_headroom_s = statistics.median(
                    cast(float, row["non_forward_headroom_s"]) for row in rows
                )
                rows = _benchmark_tabicl(
                    processor,
                    model_name,
                    tabicl,
                    dataset,
                    embedding_headroom_s,
                    device=device,
                )
                tabicl_rows.extend(rows)
                _write_csv(TABICL_OUTPUT, TABICL_FIELDS, tabicl_rows)

        if tabicl is not None:
            tabicl.clear()
        del processor
        torch.cuda.empty_cache()

    _print_summary(embedding_rows, tabicl_rows)
    print(f"\nEmbedding results written to {OUTPUT}")
    if tabicl is not None:
        print(f"TabICLv2 results written to {TABICL_OUTPUT}")


if __name__ == "__main__":
    main()
