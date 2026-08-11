# ruff: noqa: T201
"""Benchmark GPU WordPiece tokenization vs CPU fallback.

Compares the SentenceTransformer processor with GPU-native WordPiece
tokenization (via cuDF) against the CPU fallback (via model.encode()).
Uses CUDA events for synchronization-aware timing.

Usage:
    python bench_wordpiece.py
"""

import pyarrow.parquet as pq
import torch
from huggingface_hub import hf_hub_download

import sdm
import sdm.processing as sp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_path = hf_hub_download(
    repo_id="inria-soda/STRABLE-benchmark",
    filename="clear-corpus/data.parquet",
    repo_type="dataset",
)
exclude_cols = [
    "ID",
    "Google WC",
    "Joon WC v1",
    "British WC",
    "British Words",
    "Sentence Count v1",
    "Sentence Count v2",
    "Paragraphs",
    "Flesch-Reading-Ease",
    "Flesch-Kincaid-Grade-Level",
    "Automated Readability Index",
    "SMOG Readability",
    "New Dale-Chall Readability Formula",
    "CAREC",
    "CAREC_M",
    "CARES",
    "CML2RI",
]
arrow_table = pq.read_table(data_path).drop_columns(exclude_cols)
table = sdm.TableTensor.from_arrow(
    table=arrow_table,
    stypes=sdm.infer_stypes(arrow_table, text="infer"),
    device=device,
)
target_name = "BT Easiness"
table_x = table.drop_columns(target_name)

print(f"Rows: {len(table_x)}")
print(f"Text columns: {table_x.columns[sdm.Stype.text]}")
print(f"Device: {device}")
print()

processor = sp.SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    batch_size=32,
)
processor = processor.to(device)

has_gpu_tokenizer = processor._word_piece_tokenizer is not None
if not has_gpu_tokenizer:
    print("WARNING: GPU WordPiece tokenizer not available, cannot compare.")


def benchmark(label: str, num_warmup: int = 2, num_runs: int = 5) -> None:
    """Docs."""
    for _ in range(num_warmup):
        processor.transform(table_x)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    times = []
    for _ in range(num_runs):
        start.record()
        processor.transform(table_x)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) / 1000)

    mean = sum(times) / len(times)
    print(f"{label}:")
    print(f"  runs:  {times}")
    print(f"  mean:  {mean:.4f}s")
    print()


if has_gpu_tokenizer:
    # Accuracy check: compare embeddings from both paths
    emb_gpu = processor.transform(table_x)

    saved = processor._word_piece_tokenizer
    processor._word_piece_tokenizer = None
    emb_cpu = processor.transform(table_x)
    processor._word_piece_tokenizer = saved

    max_diff = (emb_gpu.numerical - emb_cpu.numerical).abs().max().item()
    mean_diff = (emb_gpu.numerical - emb_cpu.numerical).abs().mean().item()
    print("Embedding accuracy (GPU vs CPU):")
    print(f"  max abs diff:  {max_diff:.6f}")
    print(f"  mean abs diff: {mean_diff:.6f}")
    print()

    # Benchmarks
    benchmark("GPU WordPiece tokenization")

    processor._word_piece_tokenizer = None
    benchmark("CPU fallback (model.encode)")
    processor._word_piece_tokenizer = saved
else:
    benchmark("CPU fallback (model.encode)")
