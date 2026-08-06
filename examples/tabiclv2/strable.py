import argparse

import pyarrow.parquet as pq
import torch
from huggingface_hub import hf_hub_download

import sdm
from sdm.processing import TFIDF, StypeDispatch

parser = argparse.ArgumentParser()
parser.add_argument("--disable-text", action="store_true")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_path = hf_hub_download(
    repo_id="inria-soda/STRABLE-benchmark",
    filename="clear-corpus/data.parquet",
    repo_type="dataset",
)
# Drop the pre-computed readability scores (Flesch, SMOG, word counts, ...):
# they are derived from the passage text, so keeping them would make the text
# features redundant instead of the primary signal.
readability_columns = [
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
arrow_table = pq.read_table(data_path).drop_columns(readability_columns)
table = sdm.TableTensor.from_arrow(
    table=arrow_table,
    stypes=sdm.infer_stypes(arrow_table, with_text=True),
    device=device,
)
target_name = "BT Easiness"
context_size = int(0.8 * len(table))
context = table[:context_size]
query = table[context_size:]
ground_truth = query.select_columns(target_name)


model = sdm.models.TabICLv2(device=device)
recipe = model.default_recipe()
recipe.features = (
    StypeDispatch(
        text=TFIDF(
            ngram_range=(4, 6),
            max_features=256,
        )
    )
    + recipe.features
)

with torch.amp.autocast(device.type, torch.float16, enabled=table.is_cuda):
    model.fit(
        x=context.drop_columns(target_name),
        y=context[:, target_name],
        recipe=recipe,
    )
    prediction = model.predict(query.drop_columns(target_name))

rmse = (prediction - ground_truth).pow(2).mean().sqrt()
mae = (prediction - ground_truth).abs().mean()

print(f"RMSE: {rmse:>8.3f} MAE: {mae:>8.3f}")
