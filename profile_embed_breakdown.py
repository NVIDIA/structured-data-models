# ruff: noqa: T201, D103, RUF001
"""Profile SentenceTransformer + PCA preprocessing breakdown.

Based on the STRABLE clear-corpus example. Breaks down the text
preprocessing into individual steps and times each one:

1. StringTensor reshape
2. StringTensor → PyArrow (to_arrow)
3. PyArrow → Python list (to_pylist)
4. Tokenization (CPU string → token IDs)
5. Token transfer CPU → GPU
6. Model forward pass (GPU)
7. Pooling (GPU)
8. emb.to(device, dtype)
9. PCA fit (SVD)
10. PCA transform

Usage:
    python profile_embed_breakdown.py
"""

import math
import time
from typing import cast

import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch
from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer as STModel

import sdm
from sdm import StringTensor, Stype

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
context_x = context.drop_columns(target_name)

# Load model and tokenizer.
st_model = STModel("sentence-transformers/all-MiniLM-L6-v2")
st_model = st_model.to(device)
tokenizer = st_model.tokenizer
batch_size = 32

# --- Step 1: StringTensor reshape ---
if torch.cuda.is_available():
    torch.cuda.synchronize()
t0 = time.perf_counter()

text = cast(StringTensor, context_x.text.movedim(-1, 0).reshape(-1))

t1 = time.perf_counter()
t_reshape = t1 - t0

n_strings = text.numel()
columns = context_x.columns[Stype.text]
n_cols = len(columns)
batch_shape = context_x.text.shape[:-1]
n_rows = math.prod(batch_shape)
embedding_dim = st_model.get_embedding_dimension()

print(f"n_strings: {n_strings} ({n_rows} rows × {n_cols} cols)")
print(f"device: {device}")
print(f"batch_size: {batch_size}")
print()

# --- Step 2: StringTensor → PyArrow ---
t0 = time.perf_counter()
array = text.to_arrow()
t_to_arrow = time.perf_counter() - t0

# --- Step 3: PyArrow → Python list ---
t0 = time.perf_counter()
if text.is_nullable:
    array = pc.fill_null(array, "")
pylist = array.to_pylist()
t_to_pylist = time.perf_counter() - t0

# --- Steps 4-7: Break down .encode() per batch ---
n_batches = math.ceil(len(pylist) / batch_size)
t_tokenize_total = 0.0
t_to_gpu_total = 0.0
t_forward_total = 0.0
t_pool_total = 0.0

all_embeddings = []

for i in range(0, len(pylist), batch_size):
    batch_texts = pylist[i : i + batch_size]

    # Step 4: Tokenization (CPU)
    t0 = time.perf_counter()
    encoded = tokenizer(
        batch_texts,
        padding=True,
        truncation=True,
        max_length=st_model.max_seq_length,
        return_tensors="pt",
    )
    t_tokenize_total += time.perf_counter() - t0

    # Step 5: Transfer tokens CPU → GPU
    t0 = time.perf_counter()
    encoded = {k: v.to(device) for k, v in encoded.items()}
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_to_gpu_total += time.perf_counter() - t0

    # Step 6: Model forward pass (GPU)
    t0 = time.perf_counter()
    with torch.no_grad():
        model_output = st_model[0].auto_model(**encoded)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_forward_total += time.perf_counter() - t0

    # Step 7: Pooling (GPU)
    t0 = time.perf_counter()
    with torch.no_grad():
        token_embeddings = model_output.last_hidden_state
        attention_mask = encoded["attention_mask"]
        input_expanded = (
            attention_mask.unsqueeze(-1)
            .expand(token_embeddings.size())
            .float()
        )
        pooled = torch.sum(token_embeddings * input_expanded, 1) / torch.clamp(
            input_expanded.sum(1), min=1e-9
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t_pool_total += time.perf_counter() - t0

    all_embeddings.append(pooled)

emb = torch.cat(all_embeddings, dim=0)

# --- Step 8: emb.to(device, dtype) ---
if torch.cuda.is_available():
    torch.cuda.synchronize()
t0 = time.perf_counter()
emb = emb.to(device=device, dtype=table.dtype)
if torch.cuda.is_available():
    torch.cuda.synchronize()
t_to_device = time.perf_counter() - t0

# --- Step 9: PCA fit (SVD) ---
if torch.cuda.is_available():
    torch.cuda.synchronize()
t0 = time.perf_counter()
mean = emb.mean(dim=0)
centered = emb - mean
U, S, Vh = torch.linalg.svd(centered, full_matrices=False)
if torch.cuda.is_available():
    torch.cuda.synchronize()
t_pca_fit = time.perf_counter() - t0

# --- Step 10: PCA transform ---
t0 = time.perf_counter()
components = Vh[:64]
projected = centered @ components.T
if torch.cuda.is_available():
    torch.cuda.synchronize()
t_pca_transform = time.perf_counter() - t0

# --- Print results ---
total = (
    t_reshape
    + t_to_arrow
    + t_to_pylist
    + t_tokenize_total
    + t_to_gpu_total
    + t_forward_total
    + t_pool_total
    + t_to_device
    + t_pca_fit
    + t_pca_transform
)


def fmt(label, step, t):
    pct = t / total * 100
    loc = (
        "CPU"
        if step in (1, 2, 3, 4)
        else "GPU"
        if step in (6, 7, 9, 10)
        else "CPU→GPU"
    )
    return f"  {step:>2}. {label:30s} {t:8.4f}s  {pct:5.1f}%  [{loc}]"


print("=" * 65)
print("Step breakdown")
print("=" * 65)
print(fmt("StringTensor reshape", 1, t_reshape))
print(fmt("to_arrow (StringTensor→Arrow)", 2, t_to_arrow))
print(fmt("to_pylist (Arrow→Python list)", 3, t_to_pylist))
print(fmt(f"Tokenize ({n_batches} batches)", 4, t_tokenize_total))
print(fmt(f"Tokens CPU→GPU ({n_batches} batches)", 5, t_to_gpu_total))
print(fmt(f"Forward pass ({n_batches} batches)", 6, t_forward_total))
print(fmt(f"Pooling ({n_batches} batches)", 7, t_pool_total))
print(fmt("emb.to(device, dtype)", 8, t_to_device))
print(fmt("PCA fit (SVD)", 9, t_pca_fit))
print(fmt("PCA transform", 10, t_pca_transform))
print("-" * 65)
print(f"  Total: {total:.4f}s")
