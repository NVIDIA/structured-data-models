"""Compare CPU and GPU ByteLevel BPE paths with DistilRoBERTa.

The benchmark runs the same ``sdm.processing.SentenceTransformer`` processor
twice: once through SentenceTransformer's native CPU tokenizer and once
through SDM's GPU-native BPE tokenizer. Timings include the complete processor
path so the CPU result includes host conversion, tokenization, and transfer.

Run with:
    PYTHONPATH=. python examples/tabiclv2/strable/benchmark_bpe.py
"""

from __future__ import annotations

import csv
import statistics
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, TypedDict, cast

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

import sdm
import sdm.processing as sp
from sdm import StringTensor, Stype, TableTensor

MODEL_NAME = "sentence-transformers/all-distilroberta-v1"
DATASET_REPO = "inria-soda/STRABLE-benchmark"
DATASETS = (
    "chocolate-bar-ratings",
    "mercari",
    "covid-clinical-trials",
    "clear-corpus",
    "financial-product-complaint",
)
BATCH_SIZE = 32
WARMUP_RUNS = 1
RUNS = 3
USE_AMP = True
OUTPUT = Path("bpe_sentence_transformer_cpu_gpu.csv")

FIELDS = (
    "model",
    "dataset",
    "run",
    "num_rows_total",
    "num_rows",
    "num_text_columns",
    "num_strings",
    "num_non_ascii_strings",
    "batch_size",
    "max_seq_length",
    "cpu_s",
    "gpu_s",
    "cpu_over_gpu_speedup",
    "token_id_matches",
    "token_id_mismatches",
    "token_id_match_rate",
    "first_token_mismatch_index",
    "max_abs_error",
    "mean_abs_error",
    "mean_cosine_similarity",
    "min_cosine_similarity",
)


class _TokenComparison(TypedDict):
    token_id_matches: int
    token_id_mismatches: int
    token_id_match_rate: float
    first_token_mismatch_index: int | None


def _load_dataset(name: str) -> tuple[pa.Table, int, list[str]]:
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

    return full_table, len(full_table), text_columns


def _count_non_ascii(
    table: pa.Table,
    text_columns: Sequence[str],
) -> int:
    count = 0
    for column in text_columns:
        values = pc.drop_null(table.column(column).combine_chunks())
        if len(values) == 0:
            continue
        char_lengths = pc.utf8_length(values)
        byte_lengths = pc.binary_length(values)
        count += pc.sum(pc.greater(byte_lengths, char_lengths)).as_py() or 0
    return count


def _run(
    processor: sp.SentenceTransformer,
    table: TableTensor,
    tokenizer: Any | None,
) -> TableTensor:
    processor._bpe_tokenizer = tokenizer
    with (
        torch.inference_mode(),
        torch.autocast(
            device_type=table.device.type,
            dtype=torch.float16,
            enabled=USE_AMP,
        ),
    ):
        return cast(TableTensor, processor(table))


def _time_call(
    function: Callable[[], TableTensor],
    device: torch.device,
) -> tuple[TableTensor, float]:
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    output = function()
    torch.cuda.synchronize(device)
    return output, time.perf_counter() - start


def _compare_embeddings(
    cpu_output: TableTensor,
    gpu_output: TableTensor,
    *,
    num_rows: int,
    num_text_columns: int,
) -> dict[str, float]:
    cpu = cpu_output.numerical.float()
    gpu = gpu_output.numerical.float()
    difference = (cpu - gpu).abs()
    cpu = cpu.reshape(num_rows, num_text_columns, -1).flatten(0, 1)
    gpu = gpu.reshape(num_rows, num_text_columns, -1).flatten(0, 1)
    cosine = F.cosine_similarity(cpu, gpu, dim=-1)
    return {
        "max_abs_error": difference.max().item(),
        "mean_abs_error": difference.mean().item(),
        "mean_cosine_similarity": cosine.mean().item(),
        "min_cosine_similarity": cosine.min().item(),
    }


def _compare_token_ids(
    processor: sp.SentenceTransformer,
    gpu_tokenizer: Any,
    text_table: TableTensor,
) -> _TokenComparison:
    text = cast(
        StringTensor,
        text_table.text.movedim(-1, 0).reshape(-1),
    )
    array = text.to_arrow()
    if text.is_nullable:
        array = pc.fill_null(array, "")
    texts = cast(list[str], array.to_pylist())

    text_series = text.to_cudf()
    if text.is_nullable:
        text_series = text_series.fillna("")
    flat_ids, raw_lengths = gpu_tokenizer.tokenize(text_series)
    gpu_ids = flat_ids.cpu().tolist()
    gpu_lengths = raw_lengths.cpu().tolist()

    model = processor._model.module
    max_content_length = gpu_tokenizer.max_length - 2
    num_matches = 0
    first_mismatch: tuple[int, int, list[int], list[int]] | None = None
    offset = 0
    for batch_start in range(0, len(texts), BATCH_SIZE):
        batch = texts[batch_start : batch_start + BATCH_SIZE]
        features = model.preprocess(cast(list[Any], batch))
        input_ids = cast(torch.Tensor, features["input_ids"])
        attention_mask = cast(torch.Tensor, features["attention_mask"])
        for row, text_index in enumerate(
            range(batch_start, batch_start + len(batch))
        ):
            raw_length = gpu_lengths[text_index]
            content_length = min(raw_length, max_content_length)
            gpu_sequence = [
                gpu_tokenizer.bos_id,
                *gpu_ids[offset : offset + content_length],
                gpu_tokenizer.eos_id,
            ]
            offset += raw_length
            cpu_sequence = input_ids[row][attention_mask[row].bool()].tolist()
            if cpu_sequence == gpu_sequence:
                num_matches += 1
                continue
            if first_mismatch is None:
                shared_length = min(len(cpu_sequence), len(gpu_sequence))
                mismatch_position = next(
                    (
                        position
                        for position in range(shared_length)
                        if cpu_sequence[position] != gpu_sequence[position]
                    ),
                    shared_length,
                )
                first_mismatch = (
                    text_index,
                    mismatch_position,
                    cpu_sequence,
                    gpu_sequence,
                )

    if first_mismatch is not None:
        text_index, position, cpu_sequence, gpu_sequence = first_mismatch
        start = max(0, position - 5)
        stop = position + 6
        print(
            f"  first token mismatch at string {text_index}, token {position}"
        )
        print(f"    text: {texts[text_index][:200]!r}")
        print(f"    CPU IDs: {cpu_sequence[start:stop]}")
        print(f"    GPU IDs: {gpu_sequence[start:stop]}")

    num_strings = len(texts)
    return {
        "token_id_matches": num_matches,
        "token_id_mismatches": num_strings - num_matches,
        "token_id_match_rate": num_matches / num_strings,
        "first_token_mismatch_index": (
            first_mismatch[0] if first_mismatch is not None else None
        ),
    }


def _benchmark_dataset(
    processor: sp.SentenceTransformer,
    gpu_tokenizer: Any,
    dataset_name: str,
    *,
    device: torch.device,
) -> list[dict[str, object]]:
    table, num_rows_total, text_columns = _load_dataset(dataset_name)
    if not text_columns:
        print("  skipped: no inferred text columns")
        return []

    text_table = TableTensor.from_arrow(
        table=table.select(text_columns),
        stypes=dict.fromkeys(text_columns, Stype.text),
        device=device,
    )
    paths = {
        "cpu": lambda: _run(processor, text_table, None),
        "gpu": lambda: _run(processor, text_table, gpu_tokenizer),
    }
    token_comparison = _compare_token_ids(
        processor,
        gpu_tokenizer,
        text_table,
    )

    for _ in range(WARMUP_RUNS):
        for path in paths.values():
            path()
        torch.cuda.synchronize(device)

    cpu_output = paths["cpu"]()
    gpu_output = paths["gpu"]()
    torch.cuda.synchronize(device)
    comparison = _compare_embeddings(
        cpu_output,
        gpu_output,
        num_rows=len(table),
        num_text_columns=len(text_columns),
    )
    del cpu_output, gpu_output

    timings: list[dict[str, float]] = []
    for run in range(RUNS):
        order = ("cpu", "gpu") if run % 2 == 0 else ("gpu", "cpu")
        run_times: dict[str, float] = {}
        for path_name in order:
            output, seconds = _time_call(paths[path_name], device)
            run_times[path_name] = seconds
            del output
        timings.append(run_times)

    non_ascii = _count_non_ascii(table, text_columns)
    rows: list[dict[str, object]] = []
    for run, timing in enumerate(timings):
        rows.append(
            {
                "model": MODEL_NAME,
                "dataset": dataset_name,
                "run": run,
                "num_rows_total": num_rows_total,
                "num_rows": len(table),
                "num_text_columns": len(text_columns),
                "num_strings": len(table) * len(text_columns),
                "num_non_ascii_strings": non_ascii,
                "batch_size": BATCH_SIZE,
                "max_seq_length": processor._model.module.max_seq_length,
                "cpu_s": timing["cpu"],
                "gpu_s": timing["gpu"],
                "cpu_over_gpu_speedup": timing["cpu"] / timing["gpu"],
                **token_comparison,
                **comparison,
            }
        )

    median_cpu = statistics.median(timing["cpu"] for timing in timings)
    median_gpu = statistics.median(timing["gpu"] for timing in timings)
    print(
        f"  token IDs {token_comparison['token_id_matches']}/"
        f"{len(table) * len(text_columns)} exact "
        f"({100 * token_comparison['token_id_match_rate']:.4f}%)"
    )
    print(
        f"  {len(table)} rows, {len(text_columns)} text columns, "
        f"CPU {median_cpu:.3f}s, GPU {median_gpu:.3f}s, "
        f"speedup {median_cpu / median_gpu:.2f}x"
    )
    print(
        f"  max embedding error {comparison['max_abs_error']:.6g}, "
        f"mean cosine {comparison['mean_cosine_similarity']:.8f}, "
        f"min cosine {comparison['min_cosine_similarity']:.8f}"
    )
    return rows


def _write_results(rows: Sequence[dict[str, object]]) -> None:
    with OUTPUT.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA")

    device = torch.device("cuda")
    processor = sp.SentenceTransformer(
        MODEL_NAME,
        batch_size=BATCH_SIZE,
    ).to(device)
    processor.eval()
    gpu_tokenizer = processor._bpe_tokenizer
    if gpu_tokenizer is None:
        raise RuntimeError(
            f"{MODEL_NAME!r} did not initialize the GPU BPE tokenizer"
        )

    rows: list[dict[str, object]] = []
    try:
        print(f"Model: {MODEL_NAME}")
        for dataset_name in DATASETS:
            print(f"\n{dataset_name}")
            rows.extend(
                _benchmark_dataset(
                    processor,
                    gpu_tokenizer,
                    dataset_name,
                    device=device,
                )
            )
            _write_results(rows)
    finally:
        processor._bpe_tokenizer = gpu_tokenizer

    print(f"\nResults written to {OUTPUT}")


if __name__ == "__main__":
    main()
