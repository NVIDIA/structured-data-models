import torch
from sdm import TableTensor, infer_stypes
from sdm.models import TabICLv2
from sklearn.datasets import load_breast_cancer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# NVIDIA inference recipe (measured on GB200; see
# examples/benchmark_tabiclv2.py for the full ablation). This exact
# configuration - TF32 + bf16 autocast + dynamic fullgraph compilation -
# measures 6.2x on large-table one-shot inference (103 -> 16.6 ms) and
# 2.8x on fit/predict serving (25.8 -> 9.0 ms). Casting the model fully
# to bf16 instead of autocast is slightly faster still, and
# mode="reduce-overhead" reaches ~10x on small launch-bound tables. Keep
# dynamic=True: in-context learning sees a new shape per table, and
# static compilation recompiles every time (~5 s/table).
torch.set_float32_matmul_precision("high")

df = load_breast_cancer(as_frame=True).frame

table = TableTensor.from_pandas(
    df=df,
    stypes=infer_stypes(df, overrides={"target": "categorical"}),
    device=device,
)
model = TabICLv2(device=device)
# Compilation pays off when the model is called repeatedly (the first
# call per graph spends ~a minute compiling); for one-off exploratory
# runs, skip it and keep the bf16 autocast below (~3.4x by itself).
# Compile the submodels your workload uses (classification here).
if table.is_cuda:
    model.cls_model.compile(fullgraph=True, dynamic=True)

# Default in-context learning forward pass:
with torch.amp.autocast(device.type, torch.bfloat16, enabled=table.is_cuda):
    model(
        x=table.drop_columns("target"),
        y=table[:300, "target"],
        num_estimators=2,
    )

# Fit + Predict forward pass via key/value caching for fast inference.
# Caching and cached inference must share the same dtype context: fitting
# under autocast and predicting outside it raises a ValueError.
with torch.amp.autocast(device.type, torch.bfloat16, enabled=table.is_cuda):
    model.fit(
        x=table[:300].drop_columns("target"),
        y=table[:300, "target"],
        num_estimators=2,
    )
    model.predict(
        x=table[300:].drop_columns("target"),
    )
    model.clear()
