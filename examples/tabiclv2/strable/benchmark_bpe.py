# ruff: noqa: BLE001
"""Benchmark the current CPU SentenceTransformer path for BPE models.

The default run is a pilot over five STRABLE datasets chosen to span text
length, text-column count, and table size. It measures the existing SDM data
round trip, SentenceTransformer preprocessing, token transfer, and model
forward time without changing the tokenizer implementation.

Examples:
    python examples/tabiclv2/strable/benchmark_bpe.py
    python examples/tabiclv2/strable/benchmark_bpe.py --tabicl
    python examples/tabiclv2/strable/benchmark_bpe.py \
        --datasets clear-corpus mercari
    python examples/tabiclv2/strable/benchmark_bpe.py --all-datasets
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import statistics
import time
import traceback
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from typing import Any, cast

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import sentence_transformers
import torch
from huggingface_hub import HfApi, hf_hub_download
from torch import Tensor

import sdm
import sdm.processing as sp
from sdm import StringTensor, Stype, TableTensor
from sdm.processing import Processor

DATASET_REPO = "inria-soda/STRABLE-benchmark"
DEFAULT_DATASETS = (
    "chocolate-bar-ratings",
    "mercari",
    "covid-clinical-trials",
    "clear-corpus",
    "financial-product-complaint",
)
DEFAULT_MODELS = (
    "sentence-transformers/all-distilroberta-v1",
    "nomic-ai/modernbert-embed-base",
    "Qwen/Qwen3-Embedding-0.6B",
)

EMBEDDING_FIELDS = (
    "model",
    "dataset",
    "run",
    "tokenizer_model",
    "pre_tokenizer",
    "native_max_seq_length",
    "max_seq_length",
    "batch_size",
    "num_rows_total",
    "num_rows",
    "num_text_columns",
    "num_strings",
    "num_non_null_strings",
    "num_non_ascii_strings",
    "mean_chars",
    "p95_chars",
    "max_chars",
    "reshape_s",
    "to_arrow_s",
    "to_pylist_s",
    "preprocess_s",
    "transfer_wall_s",
    "transfer_gpu_s",
    "forward_gpu_s",
    "encode_wall_s",
    "total_wall_s",
    "non_forward_headroom_s",
    "non_forward_headroom_share",
    "max_speedup_bound",
    "strings_per_s",
)

TABICL_FIELDS = (
    "model",
    "dataset",
    "run",
    "num_rows",
    "num_text_columns",
    "fit_s",
    "predict_s",
    "total_s",
    "embedding_headroom_s",
    "end_to_end_headroom_share",
    "max_speedup_bound",
)


class _ModelReference(torch.nn.Module):
    """Keep one SentenceTransformer model across copied recipes."""

    def __init__(
        self,
        module: sentence_transformers.SentenceTransformer,
    ) -> None:
        super().__init__()
        self.module = module

    def __deepcopy__(self, memo: dict[int, Any]) -> _ModelReference:
        return type(self)(self.module)


class _CPUTextEmbedding(Processor):
    """Benchmark-only copy of the current CPU SentenceTransformer path."""

    handles_stypes = frozenset({Stype.text})
    requires_fit = False

    def __init__(
        self,
        model: sentence_transformers.SentenceTransformer,
        *,
        batch_size: int,
    ) -> None:
        super().__init__()
        embedding_dim = model.get_embedding_dimension()
        assert isinstance(embedding_dim, int)
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        self.model = _ModelReference(model)

    def _transform(self, table: TableTensor) -> TableTensor:
        columns = table.columns[Stype.text]
        batch_shape = table.text.shape[:-1]
        output_columns = tuple(
            f"{column}__emb{i}"
            for column in columns
            for i in range(self.embedding_dim)
        )

        text = cast(StringTensor, table.text.movedim(-1, 0).reshape(-1))
        array = text.to_arrow()
        if text.is_nullable:
            array = pc.fill_null(array, "")
        emb = self.model.module.encode(
            array.to_pylist(),
            show_progress_bar=False,
            convert_to_tensor=True,
            device=str(table.device),
            batch_size=self.batch_size,
        )
        assert isinstance(emb, Tensor)
        numerical = (
            emb.to(device=table.device, dtype=table.dtype)
            .reshape(len(columns), *batch_shape, self.embedding_dim)
            .movedim(0, -2)
            .reshape(*batch_shape, len(output_columns))
        )
        embedded = TableTensor(
            columns={Stype.numerical: output_columns},
            numerical=numerical,
        )
        return cast(
            TableTensor,
            torch.cat([table.drop_stypes(Stype.text), embedded], dim=-1),
        )


class _CsvSink:
    def __init__(self, path: Path, fields: Sequence[str]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = path.open("w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=fields)
        self._writer.writeheader()

    def write(self, row: dict[str, object]) -> None:
        self._writer.writerow(row)
        self._file.flush()

    def close(self) -> None:
        self._file.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    dataset_group = parser.add_mutually_exclusive_group()
    dataset_group.add_argument(
        "--datasets",
        nargs="+",
        help="STRABLE dataset names. Defaults to the five-dataset pilot.",
    )
    dataset_group.add_argument(
        "--all-datasets",
        action="store_true",
        help="Run every STRABLE dataset containing inferred text columns.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_MODELS),
        help="SentenceTransformer BPE model names.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=2048,
        help=(
            "Evenly sample at most this many rows per table; 0 uses all rows."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=512,
        help=(
            "Cap, but never extend, each model's configured maximum; 0 uses "
            "its native maximum."
        ),
    )
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use float16 autocast, matching the TabICLv2 example.",
    )
    parser.add_argument(
        "--tabicl",
        action="store_true",
        help=(
            "Also time TabICLv2 fit and predict with the same CPU embedding "
            "path."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("bpe_sentence_transformer.csv"),
    )
    return parser.parse_args()


def _all_datasets() -> list[str]:
    files = HfApi().list_repo_files(DATASET_REPO, repo_type="dataset")
    return sorted(
        path.removesuffix("/data.parquet")
        for path in files
        if path.count("/") == 1 and path.endswith("/data.parquet")
    )


def _load_dataset(
    name: str,
    *,
    max_rows: int,
) -> tuple[pa.Table, pa.Table, dict[str, Stype], list[str]]:
    data_path = hf_hub_download(
        repo_id=DATASET_REPO,
        filename=f"{name}/data.parquet",
        repo_type="dataset",
    )
    full_table = pq.read_table(data_path)
    stypes = {
        column: Stype(stype)
        for column, stype in sdm.infer_stypes(
            full_table,
            text="infer",
        ).items()
    }
    text_columns = [
        column for column, stype in stypes.items() if stype == Stype.text
    ]

    table = full_table
    if max_rows > 0 and len(table) > max_rows:
        indices = np.linspace(
            0,
            len(table) - 1,
            num=max_rows,
            dtype=np.int64,
        )
        table = table.take(pa.array(indices))
    return full_table, table, stypes, text_columns


def _text_stats(
    table: pa.Table,
    text_columns: Sequence[str],
) -> dict[str, int | float]:
    char_lengths: list[pa.Array] = []
    byte_lengths: list[pa.Array] = []
    for column in text_columns:
        array = pc.drop_null(table.column(column).combine_chunks())
        if len(array) == 0:
            continue
        char_lengths.append(pc.utf8_length(array))
        byte_lengths.append(pc.binary_length(array))

    if not char_lengths:
        return {
            "num_strings": len(table) * len(text_columns),
            "num_non_null_strings": 0,
            "num_non_ascii_strings": 0,
            "mean_chars": 0,
            "p95_chars": 0,
            "max_chars": 0,
        }

    chars = pa.concat_arrays(char_lengths)
    byte_count = pa.concat_arrays(byte_lengths)
    non_ascii = pc.sum(pc.greater(byte_count, chars)).as_py() or 0
    p95_chars = pc.quantile(chars, q=0.95)[0].as_py() or 0
    return {
        "num_strings": len(table) * len(text_columns),
        "num_non_null_strings": len(chars),
        "num_non_ascii_strings": non_ascii,
        "mean_chars": round(pc.mean(chars).as_py() or 0, 2),
        "p95_chars": round(p95_chars, 2),
        "max_chars": pc.max(chars).as_py() or 0,
    }


def _sync(device: torch.device) -> None:
    torch.cuda.synchronize(device)


def _time_call(
    function: Callable[[], Any],
    *,
    device: torch.device,
) -> tuple[Any, float]:
    _sync(device)
    start = time.perf_counter()
    out = function()
    _sync(device)
    return out, time.perf_counter() - start


def _event_seconds(
    events: Sequence[tuple[torch.cuda.Event, torch.cuda.Event]],
) -> float:
    return sum(start.elapsed_time(end) for start, end in events) / 1000


def _to_pylist(text: StringTensor, array: pa.Array) -> list[str]:
    if text.is_nullable:
        array = pc.fill_null(array, "")
    return array.to_pylist()


@contextmanager
def _profile_encode(
    model: sentence_transformers.SentenceTransformer,
    *,
    device: torch.device,
) -> Iterator[dict[str, Any]]:
    """Instrument the exact methods called by SentenceTransformer.encode."""
    state: dict[str, Any] = {}
    original_preprocess = model.preprocess
    original_forward = model.forward
    model_any = cast(Any, model)
    encode_module = importlib.import_module(model.encode.__module__)
    original_batch_to_device = encode_module.batch_to_device

    def preprocess(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        out = original_preprocess(*args, **kwargs)
        state["preprocess_s"] += time.perf_counter() - start
        return out

    def batch_to_device(*args: Any, **kwargs: Any) -> Any:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        start = time.perf_counter()
        out = original_batch_to_device(*args, **kwargs)
        state["transfer_wall_s"] += time.perf_counter() - start
        end_event.record()
        state["transfer_events"].append((start_event, end_event))
        return out

    def forward(*args: Any, **kwargs: Any) -> Any:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        out = original_forward(*args, **kwargs)
        end_event.record()
        state["forward_events"].append((start_event, end_event))
        return out

    model_any.preprocess = preprocess
    model_any.forward = forward
    encode_module.batch_to_device = batch_to_device
    try:
        yield state
    finally:
        model_any.preprocess = original_preprocess
        model_any.forward = original_forward
        encode_module.batch_to_device = original_batch_to_device


def _materialize_text(
    table: TableTensor,
) -> tuple[StringTensor, list[str]]:
    text = cast(StringTensor, table.text.movedim(-1, 0).reshape(-1))
    array = text.to_arrow()
    if text.is_nullable:
        array = pc.fill_null(array, "")
    return text, array.to_pylist()


def _encode(
    model: sentence_transformers.SentenceTransformer,
    texts: list[str],
    *,
    batch_size: int,
    device: torch.device,
    amp: bool,
) -> Tensor:
    with torch.amp.autocast(
        device.type,
        torch.float16,
        enabled=amp,
    ):
        emb = model.encode(
            texts,
            show_progress_bar=False,
            convert_to_tensor=True,
            device=str(device),
            batch_size=batch_size,
        )
    assert isinstance(emb, Tensor)
    return emb


def _tokenizer_types(
    model: sentence_transformers.SentenceTransformer,
) -> tuple[str, str]:
    tokenizer = model.tokenizer
    backend = getattr(tokenizer, "backend_tokenizer", None)
    tokenizer_model = type(backend.model).__name__ if backend else "unknown"
    pre_tokenizer = (
        type(backend.pre_tokenizer).__name__ if backend else "unknown"
    )
    return tokenizer_model, pre_tokenizer


def _profile_embedding(
    model_name: str,
    model: sentence_transformers.SentenceTransformer,
    native_max_length: int | None,
    dataset_name: str,
    full_table: pa.Table,
    table: pa.Table,
    text_columns: Sequence[str],
    *,
    args: argparse.Namespace,
    device: torch.device,
) -> list[dict[str, object]]:
    text_table = table.select(text_columns)
    tensor = TableTensor.from_arrow(
        table=text_table,
        stypes=dict.fromkeys(text_columns, Stype.text),
        device=device,
    )
    stats = _text_stats(table, text_columns)
    tokenizer_model, pre_tokenizer = _tokenizer_types(model)
    if tokenizer_model != "BPE":
        raise ValueError(
            f"Expected a BPE tokenizer for {model_name!r}, got "
            f"{tokenizer_model!r}"
        )

    for _ in range(args.warmup_runs):
        _, texts = _materialize_text(tensor)
        emb = _encode(
            model,
            texts,
            batch_size=args.batch_size,
            device=device,
            amp=args.amp,
        )
        _sync(device)
        del emb

    rows: list[dict[str, object]] = []
    with _profile_encode(model, device=device) as profile:
        for run in range(args.runs):
            profile.clear()
            profile.update(
                preprocess_s=0.0,
                transfer_wall_s=0.0,
                transfer_events=[],
                forward_events=[],
            )

            text, reshape_s = _time_call(
                lambda: cast(
                    StringTensor,
                    tensor.text.movedim(-1, 0).reshape(-1),
                ),
                device=device,
            )
            array, to_arrow_s = _time_call(text.to_arrow, device=device)
            texts, to_pylist_s = _time_call(
                partial(_to_pylist, text, array),
                device=device,
            )
            emb, encode_wall_s = _time_call(
                partial(
                    _encode,
                    model,
                    texts,
                    batch_size=args.batch_size,
                    device=device,
                    amp=args.amp,
                ),
                device=device,
            )
            del emb

            transfer_gpu_s = _event_seconds(profile["transfer_events"])
            forward_gpu_s = _event_seconds(profile["forward_events"])
            total_wall_s = reshape_s + to_arrow_s + to_pylist_s + encode_wall_s
            headroom_s = max(total_wall_s - forward_gpu_s, 0.0)
            headroom_share = headroom_s / total_wall_s
            max_speedup = (
                total_wall_s / forward_gpu_s if forward_gpu_s > 0 else math.inf
            )
            rows.append(
                {
                    "model": model_name,
                    "dataset": dataset_name,
                    "run": run,
                    "tokenizer_model": tokenizer_model,
                    "pre_tokenizer": pre_tokenizer,
                    "native_max_seq_length": native_max_length,
                    "max_seq_length": model.max_seq_length,
                    "batch_size": args.batch_size,
                    "num_rows_total": len(full_table),
                    "num_rows": len(table),
                    "num_text_columns": len(text_columns),
                    **stats,
                    "reshape_s": reshape_s,
                    "to_arrow_s": to_arrow_s,
                    "to_pylist_s": to_pylist_s,
                    "preprocess_s": profile["preprocess_s"],
                    "transfer_wall_s": profile["transfer_wall_s"],
                    "transfer_gpu_s": transfer_gpu_s,
                    "forward_gpu_s": forward_gpu_s,
                    "encode_wall_s": encode_wall_s,
                    "total_wall_s": total_wall_s,
                    "non_forward_headroom_s": headroom_s,
                    "non_forward_headroom_share": headroom_share,
                    "max_speedup_bound": max_speedup,
                    "strings_per_s": stats["num_strings"] / total_wall_s,
                }
            )

    return rows


def _load_target_name(dataset_name: str) -> str:
    config_path = hf_hub_download(
        repo_id=DATASET_REPO,
        filename=f"{dataset_name}/config.json",
        repo_type="dataset",
    )
    with Path(config_path).open() as file:
        return cast(str, json.load(file)["target_name"])


def _benchmark_tabicl(
    model_name: str,
    embedding_model: sentence_transformers.SentenceTransformer,
    tabicl: sdm.models.TabICLv2,
    dataset_name: str,
    table: pa.Table,
    stypes: dict[str, Stype],
    text_columns: Sequence[str],
    embedding_headroom_s: float,
    *,
    args: argparse.Namespace,
    device: torch.device,
) -> list[dict[str, object]]:
    target_name = _load_target_name(dataset_name)
    tensor = TableTensor.from_arrow(
        table=table,
        stypes=stypes,
        device=device,
    )
    rows: list[dict[str, object]] = []

    for run in range(args.runs):
        generator = torch.Generator(device=device).manual_seed(args.seed)
        perm = torch.randperm(len(tensor), generator=generator, device=device)
        context_size = int(0.8 * len(tensor))
        context = tensor[perm[:context_size]]
        query = tensor[perm[context_size:]]

        text_processor = _CPUTextEmbedding(
            embedding_model,
            batch_size=args.batch_size,
        )
        recipe = tabicl.default_recipe().prepend_features(
            sp.StypeDispatch(
                text=sp.Sequential(
                    text_processor,
                    sp.PCA(num_components=64),
                )
            )
        )
        _, fit_s = _time_call(
            partial(
                _fit_tabicl,
                tabicl,
                context,
                target_name,
                recipe,
                generator,
                amp=args.amp,
                device=device,
            ),
            device=device,
        )
        _, predict_s = _time_call(
            partial(
                _predict_tabicl,
                tabicl,
                query,
                target_name,
                amp=args.amp,
                device=device,
            ),
            device=device,
        )
        total_s = fit_s + predict_s
        removable_s = min(embedding_headroom_s, total_s)
        rows.append(
            {
                "model": model_name,
                "dataset": dataset_name,
                "run": run,
                "num_rows": len(table),
                "num_text_columns": len(text_columns),
                "fit_s": fit_s,
                "predict_s": predict_s,
                "total_s": total_s,
                "embedding_headroom_s": embedding_headroom_s,
                "end_to_end_headroom_share": removable_s / total_s,
                "max_speedup_bound": (
                    total_s / (total_s - removable_s)
                    if removable_s < total_s
                    else math.inf
                ),
            }
        )
    return rows


def _fit_tabicl(
    tabicl: sdm.models.TabICLv2,
    context: TableTensor,
    target_name: str,
    recipe: sdm.Recipe,
    generator: torch.Generator,
    *,
    amp: bool,
    device: torch.device,
) -> None:
    with torch.amp.autocast(
        device.type,
        torch.float16,
        enabled=amp,
    ):
        tabicl.fit(
            x=context.drop_columns(target_name),
            y=context[:, target_name],
            recipe=recipe,
            generator=generator,
        )


def _predict_tabicl(
    tabicl: sdm.models.TabICLv2,
    query: TableTensor,
    target_name: str,
    *,
    amp: bool,
    device: torch.device,
) -> TableTensor:
    with torch.amp.autocast(
        device.type,
        torch.float16,
        enabled=amp,
    ):
        return tabicl.predict(query.drop_columns(target_name))


def _print_summary(
    embedding_rows: Sequence[dict[str, object]],
    tabicl_rows: Sequence[dict[str, object]],
) -> None:
    print("\nEmbedding-path summary (medians across runs and datasets)")
    print(
        "model | cases | total (s) | preprocess (s) | "
        "non-forward bound | max speedup bound"
    )
    for model_name in dict.fromkeys(row["model"] for row in embedding_rows):
        rows = [row for row in embedding_rows if row["model"] == model_name]
        total = statistics.median(
            cast(float, row["total_wall_s"]) for row in rows
        )
        preprocess = statistics.median(
            cast(float, row["preprocess_s"]) for row in rows
        )
        headroom = statistics.median(
            cast(float, row["non_forward_headroom_share"]) for row in rows
        )
        speedup = statistics.median(
            cast(float, row["max_speedup_bound"]) for row in rows
        )
        print(
            f"{model_name} | {len(rows)} | {total:.3f} | {preprocess:.3f} | "
            f"{headroom:.1%} | {speedup:.2f}x"
        )

    print(
        "\nThe non-forward figure is an optimistic ceiling: it assumes every "
        "part of the embedding path except transformer forward becomes free."
    )
    if not tabicl_rows:
        return

    print("\nTabICLv2 summary (medians across runs and datasets)")
    print("model | cases | total (s) | end-to-end bound | max speedup bound")
    for model_name in dict.fromkeys(row["model"] for row in tabicl_rows):
        rows = [row for row in tabicl_rows if row["model"] == model_name]
        total = statistics.median(cast(float, row["total_s"]) for row in rows)
        headroom = statistics.median(
            cast(float, row["end_to_end_headroom_share"]) for row in rows
        )
        speedup = statistics.median(
            cast(float, row["max_speedup_bound"]) for row in rows
        )
        print(
            f"{model_name} | {len(rows)} | {total:.3f} | "
            f"{headroom:.1%} | {speedup:.2f}x"
        )


def main() -> None:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA")

    device = torch.device("cuda")
    datasets = (
        _all_datasets()
        if args.all_datasets
        else args.datasets or list(DEFAULT_DATASETS)
    )
    tabicl_path = args.output.with_name(
        f"{args.output.stem}_tabicl{args.output.suffix}"
    )
    embedding_sink = _CsvSink(args.output, EMBEDDING_FIELDS)
    tabicl_sink = _CsvSink(tabicl_path, TABICL_FIELDS) if args.tabicl else None
    embedding_rows: list[dict[str, object]] = []
    tabicl_rows: list[dict[str, object]] = []

    try:
        for model_name in args.models:
            print(f"\nLoading {model_name}")
            try:
                embedding_model = (
                    sentence_transformers.SentenceTransformer(model_name)
                    .to(device)
                    .eval()
                )
                native_max_length = embedding_model.max_seq_length
                if args.max_seq_length > 0:
                    embedding_model.max_seq_length = (
                        args.max_seq_length
                        if native_max_length is None
                        else min(native_max_length, args.max_seq_length)
                    )
                tabicl = (
                    sdm.models.TabICLv2(device=device) if args.tabicl else None
                )
            except Exception:
                traceback.print_exc()
                continue

            for dataset_name in datasets:
                print(f"\n{model_name} / {dataset_name}")
                try:
                    full_table, table, stypes, text_columns = _load_dataset(
                        dataset_name,
                        max_rows=args.max_rows,
                    )
                    if not text_columns:
                        print("  skipped: no inferred text columns")
                        continue

                    rows = _profile_embedding(
                        model_name,
                        embedding_model,
                        native_max_length,
                        dataset_name,
                        full_table,
                        table,
                        text_columns,
                        args=args,
                        device=device,
                    )
                    for row in rows:
                        embedding_sink.write(row)
                    embedding_rows.extend(rows)

                    headroom_s = statistics.median(
                        cast(float, row["non_forward_headroom_s"])
                        for row in rows
                    )
                    median_headroom = statistics.median(
                        cast(float, row["non_forward_headroom_share"])
                        for row in rows
                    )
                    print(
                        f"  {len(table)} rows, {len(text_columns)} text "
                        f"columns; median non-forward bound "
                        f"{median_headroom:.1%}"
                    )

                    if tabicl is not None and tabicl_sink is not None:
                        rows = _benchmark_tabicl(
                            model_name,
                            embedding_model,
                            tabicl,
                            dataset_name,
                            table,
                            stypes,
                            text_columns,
                            headroom_s,
                            args=args,
                            device=device,
                        )
                        for row in rows:
                            tabicl_sink.write(row)
                        tabicl_rows.extend(rows)
                except Exception:
                    traceback.print_exc()

            del embedding_model
            if tabicl is not None:
                del tabicl
            torch.cuda.empty_cache()
    finally:
        embedding_sink.close()
        if tabicl_sink is not None:
            tabicl_sink.close()

    _print_summary(embedding_rows, tabicl_rows)
    print(f"\nEmbedding results written to {args.output}")
    if args.tabicl:
        print(f"TabICLv2 results written to {tabicl_path}")


if __name__ == "__main__":
    main()
