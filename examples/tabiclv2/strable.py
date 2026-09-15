# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TabICLv2 on STRABLE example.

Predict CLEAR Corpus readability with TabICLv2 and optional text features.
Text features are either embeded via a Sentence Transformer followed by PCA, or
encoded via character n-gram TF-IDF.
"""

import argparse

import pyarrow.parquet as pq
import torch
from huggingface_hub import hf_hub_download

import sdm
import sdm.processing as sp

parser = argparse.ArgumentParser()
parser.add_argument(
    "--text-processor",
    choices=("embed", "tfidf", "none"),
    default="embed",
)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

torch.manual_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_path = hf_hub_download(
    repo_id="inria-soda/STRABLE-benchmark",
    filename="clear-corpus/data.parquet",
    repo_type="dataset",
)
target_name = "BT Easiness"

arrow_table = pq.read_table(data_path)
table = sdm.TableTensor.from_arrow(
    table=arrow_table,
    stypes=sdm.infer_stypes(
        arrow_table,
        text="off" if args.text_processor == "none" else "infer",
    ),
    device=device,
)
table = table[torch.randperm(table.size(0), device=device)]
context, query = table.split(int(0.8 * table.size(0)))

model = sdm.models.TabICLv2(device=device)
recipe = model.default_recipe()
if args.text_processor != "none":
    if args.text_processor == "tfidf":
        text_processor = sp.TFIDF(ngram_range=(4, 6), max_features=256)
    else:
        text_processor = [
            sp.SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2"),
            sp.PCA(64),
        ]
    recipe.prepend_features(sp.StypeDispatch(text=text_processor))


with torch.amp.autocast(device.type, torch.float16, enabled=table.is_cuda):
    pred = model(
        x_context=context.drop_columns(target_name),
        y_context=context[:, target_name],
        x_query=query.drop_columns(target_name),
        recipe=recipe,
        num_estimators=8,
    ).numerical.mean(dim=-1)

y_query = query[:, target_name].numerical.squeeze(-1)
rmse = (pred - y_query).pow(2).mean().sqrt()
mae = (pred - y_query).abs().mean()
print(f"RMSE: {rmse:.3f}, MAE: {mae:.3f}")
