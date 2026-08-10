# ruff: noqa: T201
"""Profile the full TabICLv2 + SentenceTransformer pipeline.

Based on the STRABLE clear-corpus example. Wraps fit + predict in
torch.profiler to produce a Chrome trace showing where time is spent
across the full pipeline (SentenceTransformer, PCA, TabICLv2).

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
ground_truth = query[:, target_name].numerical.squeeze(-1)

model = sdm.models.TabICLv2(device=device)
recipe = model.default_recipe()
text_processor = sp.Sequential(
    sp.SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2"),
    sp.PCA(num_components=64),
)
recipe.prepend_features(sp.StypeDispatch(text=text_processor))

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
        record_shapes=True,
    ) as prof,
    torch.amp.autocast(device.type, torch.float16, enabled=table.is_cuda),
):
    model.fit(
        x=context.drop_columns(target_name),
        y=context[:, target_name],
        recipe=recipe,
        generator=generator,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    prediction = model.predict(query.drop_columns(target_name)).numerical
    prediction = prediction.mean(dim=-1)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

prof.export_chrome_trace(args.output)

rmse = (prediction - ground_truth).pow(2).mean().sqrt()
mae = (prediction - ground_truth).abs().mean()
print(f"RMSE: {rmse:.3f}, MAE: {mae:.3f}")

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
