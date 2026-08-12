# ruff: noqa: T201
"""Benchmark GPU BPE tokenization vs CPU fallback.

Compares the SentenceTransformer processor with GPU-native BPE
tokenization (via cuDF BytePairEncoder) against the CPU fallback
(via model.encode()). Uses CUDA events for synchronization-aware timing.

Usage:
    python sdm/processing/text/bench_bpe.py
"""

import pyarrow.parquet as pq
import torch
from huggingface_hub import hf_hub_download

import sdm
import sdm.processing as sp
from sdm import StringTensor

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

MODEL_NAME = "sentence-transformers/all-distilroberta-v1"
processor = sp.SentenceTransformer(MODEL_NAME, batch_size=32)
processor = processor.to(device)

model = processor._model.module
print(f"Model: {MODEL_NAME}")
tok_type = model.tokenizer.backend_tokenizer.model.__class__.__name__
print(f"Tokenizer type: {tok_type}")
print(f"model.max_seq_length: {model.max_seq_length}")
if processor._bpe_tokenizer is not None:
    print(f"_bpe_tokenizer.max_length: {processor._bpe_tokenizer.max_length}")
print()

has_gpu_tokenizer = processor._bpe_tokenizer is not None
if not has_gpu_tokenizer:
    print("WARNING: GPU BPE tokenizer not available, cannot compare.")


def benchmark(label: str, num_warmup: int = 2, num_runs: int = 5) -> None:
    """Benchmark the processor."""
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
    # Token ID comparison
    sample_texts = [
        "Hello world",
        "The quick brown fox jumps.",
        "Machine learning is great!",
    ]
    sample_tensor = StringTensor.from_list(sample_texts, device=device)

    # GPU tokenization
    bpe = processor._bpe_tokenizer
    text_series = sample_tensor.to_cudf()
    text_series = text_series.str.replace(" ", " Ġ")
    encoded = bpe.encoder(text_series)
    gpu_tokens = encoded.str.split(" ")

    # CPU tokenization (HuggingFace)
    hf_tokenizer = model.tokenizer

    print("Token ID comparison (GPU cuDF vs CPU HuggingFace):")
    for i, text in enumerate(sample_texts):
        gpu_subtokens = list(gpu_tokens.iloc[i])
        cpu_result = hf_tokenizer(text, add_special_tokens=False)
        cpu_ids = cpu_result["input_ids"]
        cpu_subtokens = hf_tokenizer.convert_ids_to_tokens(cpu_ids)
        print(f"  '{text}'")
        print(f"    GPU subtokens: {gpu_subtokens}")
        print(f"    CPU subtokens: {cpu_subtokens}")
        print(f"    CPU ids: {cpu_ids}")
    print()

    # NaN check
    emb_gpu = processor.transform(table_x)
    num_nan_gpu = emb_gpu.numerical.isnan().sum().item()

    saved = processor._bpe_tokenizer
    processor._bpe_tokenizer = None
    emb_cpu = processor.transform(table_x)
    processor._bpe_tokenizer = saved

    num_nan_cpu = emb_cpu.numerical.isnan().sum().item()
    print(
        f"NaN in GPU embeddings: {num_nan_gpu} / {emb_gpu.numerical.numel()}"
    )
    print(
        f"NaN in CPU embeddings: {num_nan_cpu} / {emb_cpu.numerical.numel()}"
    )
    print()

    # Benchmarks
    benchmark("GPU BPE tokenization")

    processor._bpe_tokenizer = None
    benchmark("CPU fallback (model.encode)")
    processor._bpe_tokenizer = saved
else:
    benchmark("CPU fallback (model.encode)")
