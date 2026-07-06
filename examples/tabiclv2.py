import torch
from sdm import TableTensor, infer_stypes
from sdm.models import TabICLv2
from sklearn.datasets import load_breast_cancer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# NVIDIA inference recipe (measured on GB200; see
# examples/benchmark_tabiclv2.py for the full ablation): TF32 matmuls are
# a free ~1.4x for fp32, bf16 gives ~4x at half the memory, and compiling
# the submodels adds up to ~7x (or ~10x on small launch-bound tables with
# mode="reduce-overhead"). Keep dynamic=True: in-context learning sees a
# new shape per table, and static compilation recompiles every time.
torch.set_float32_matmul_precision("high")

df = load_breast_cancer(as_frame=True).frame

table = TableTensor.from_pandas(
    df=df,
    stypes=infer_stypes(df, overrides={"target": "categorical"}),
    device=device,
)
model = TabICLv2(device=device)
if table.is_cuda:
    model.cls_model.compile(fullgraph=True, dynamic=True)
    model.reg_model.compile(fullgraph=True, dynamic=True)

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
