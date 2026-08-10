# ruff: noqa: T201
"""Profile SentenceTransformer tokenization vs forward pass separately.

Based on the STRABLE clear-corpus example. Breaks down the embed step
into tokenization (CPU string processing) and model forward pass
(PyTorch ops), and profiles each independently.

Usage:
    python profile_embed_breakdown.py
    python profile_embed_breakdown.py --output trace.json
"""

import argparse
import time

import pyarrow.parquet as pq
import torch
from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer as STModel

import sdm

parser = argparse.ArgumentParser()
parser.add_argument(
    "--output",
    default="profile_embed_trace.json",
    help="Output path for the Chrome trace (default: profile_embed_trace.json).",
)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = torch.Generator(device=device).manual_seed(42)

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
    stypes=sdm.infer_stypes(arrow_table, with_text=True),
    device=device,
)
target_name = "BT Easiness"
num_rows = len(table)
perm = torch.randperm(num_rows, generator=generator, device=device)
context_size = int(0.8 * num_rows)
context = table[perm[:context_size]]

# Extract text cells as a flat list of strings (same as SentenceTransformer
# processor does internally).
context_x = context.drop_columns(target_name)
stypes = sdm.infer_stypes(arrow_table, with_text=True)
text_cols = [col for col, s in stypes.items() if s == sdm.Stype.text]
texts = []
for col in text_cols:
    col_idx = list(context_x.columns[sdm.Stype.text]).index(col)
    arr = context_x.text[..., col_idx].reshape(-1).to_arrow().to_pylist()
    texts.extend(s if s is not None else "" for s in arr)

print(f"Total text cells: {len(texts)}")
print(f"Text columns: {text_cols}")
print(f"Device: {device}")

# Load the model and tokenizer.
st_model = STModel("sentence-transformers/all-MiniLM-L6-v2")
st_model = st_model.to(device)
tokenizer = st_model.tokenizer

# --- Profile tokenization + forward pass together ---
if torch.cuda.is_available():
    torch.cuda.synchronize()

with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        *(
            [torch.profiler.ProfilerActivity.CUDA]
            if torch.cuda.is_available()
            else []
        ),
    ],
    record_shapes=True,
    with_stack=True,
) as prof:
    # Step 1: tokenization (pure CPU, shows as one block in the trace)
    t0 = time.perf_counter()
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=st_model.max_seq_length,
        return_tensors="pt",
    )
    t_tokenize = time.perf_counter() - t0

    # Step 2: forward pass (PyTorch ops, broken down in the trace)
    encoded = {k: v.to(device) for k, v in encoded.items()}
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    with torch.no_grad():
        t0 = time.perf_counter()
        output = st_model(encoded)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_forward = time.perf_counter() - t0

prof.export_chrome_trace(args.output)

print(f"\nTokenization: {t_tokenize:.3f}s")
print(f"  Input IDs shape: {encoded['input_ids'].shape}")
print(f"Forward pass: {t_forward:.3f}s")
print(f"  Embedding shape: {output['sentence_embedding'].shape}")

# --- Step 3: For comparison, time the full .encode() call ---
if torch.cuda.is_available():
    torch.cuda.synchronize()

t0 = time.perf_counter()
emb = st_model.encode(
    texts,
    show_progress_bar=False,
    convert_to_tensor=True,
    device=str(device),
    batch_size=32,
)
if torch.cuda.is_available():
    torch.cuda.synchronize()
t_encode = time.perf_counter() - t0

print(f"\nFull .encode(): {t_encode:.3f}s")
print(
    f"  Tokenization:  {t_tokenize:.3f}s ({t_tokenize / t_encode * 100:.1f}%)"
)
print(f"  Forward pass:  {t_forward:.3f}s ({t_forward / t_encode * 100:.1f}%)")
print(f"  Overhead:      {t_encode - t_tokenize - t_forward:.3f}s")

print("\n--- Forward pass ops (top 20) ---")
print(
    prof.key_averages().table(
        sort_by="cpu_time_total",
        row_limit=20,
    )
)

if torch.cuda.is_available():
    print("\n--- Forward pass CUDA ops (top 20) ---")
    print(
        prof.key_averages().table(
            sort_by="cuda_time_total",
            row_limit=20,
        )
    )

print(f"\nChrome trace written to {args.output}")
