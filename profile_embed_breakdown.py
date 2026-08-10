# ruff: noqa: T201
"""Profile SentenceTransformer + PCA preprocessing only.

Based on the STRABLE clear-corpus example. Profiles only the text
preprocessing step (SentenceTransformer + PCA), excluding TabICLv2,
to produce a smaller, focused Chrome trace.

Usage:
    python profile_embed_breakdown.py
    python profile_embed_breakdown.py --output trace.json
"""

import argparse

import pyarrow.parquet as pq
import torch
from huggingface_hub import hf_hub_download

import sdm
import sdm.processing as sp

parser = argparse.ArgumentParser()
parser.add_argument(
    "--output",
    default="profile_embed_trace.json",
    help="Output path for the Chrome trace.",
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
query = table[perm[context_size:]]

context_x = context.drop_columns(target_name)
query_x = query.drop_columns(target_name)

st = sp.SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
)
pca = sp.PCA(num_components=64)
text_processor = sp.Sequential(st, pca)
dispatch = sp.StypeDispatch(text=text_processor)

if torch.cuda.is_available():
    torch.cuda.synchronize()

with (
    torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            *(
                [torch.profiler.ProfilerActivity.CUDA]
                if torch.cuda.is_available()
                else []
            ),
        ],
    ) as prof,
    torch.amp.autocast(device.type, torch.float16, enabled=table.is_cuda),
):
    with torch.profiler.record_function("preprocess.fit_transform"):
        context_out = dispatch.fit_transform(context_x)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    with torch.profiler.record_function("preprocess.transform"):
        query_out = dispatch.transform(query_x)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

prof.export_chrome_trace(args.output)

print(f"Context shape: {context_out.numerical.shape}")
print(f"Query shape: {query_out.numerical.shape}")

print("\n--- CPU time (top 30 ops) ---")
print(
    prof.key_averages().table(
        sort_by="cpu_time_total",
        row_limit=30,
    )
)

if torch.cuda.is_available():
    print("\n--- CUDA time (top 30 ops) ---")
    print(
        prof.key_averages().table(
            sort_by="cuda_time_total",
            row_limit=30,
        )
    )

print(f"\nChrome trace written to {args.output}")
