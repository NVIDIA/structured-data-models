import torch
from sdm import TableTensor, infer_stypes
from sdm.models import TabICLv2
from sklearn.datasets import load_breast_cancer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
# Cold-start-sensitive serving can instead compile the repeated
# transformer blocks individually ("regional compilation", measured
# 16 s cold start instead of ~55 s at the same large-table latency;
# launch-bound small-table and cached-predict calls measure a few ms
# slower from per-block dispatch) and cut the rest of the compile time
# with torch.compiler.save_cache_artifacts / load_cache_artifacts;
# streams of fresh table shapes are best served with bucketed padding
# via `seqused_train` / `seqused_cols` (measured as c15-c19 in
# examples/benchmark_tabiclv2.py).
if table.is_cuda:
    model.cls_model.compile(fullgraph=True)

# Default in-context learning forward pass:
with torch.amp.autocast(device.type, torch.bfloat16, enabled=table.is_cuda):
    model(
        x=table.drop_columns("target"),
        y=table[:300, "target"],
        # TODO: Re-enable once the recipe supports classification.
        # recipe=model.default_recipe(),
        num_estimators=2,
    )

# Fit + Predict forward pass via key/value caching for fast inference.
# Caching and cached inference must share the same dtype context: fitting
# under autocast and predicting outside it raises a ValueError.
with torch.amp.autocast(device.type, torch.bfloat16, enabled=table.is_cuda):
    model.fit(
        x=table[:300].drop_columns("target"),
        y=table[:300, "target"],
        # TODO: Re-enable once the recipe supports classification.
        # recipe=model.default_recipe(),
        num_estimators=2,
    )
    model.predict(
        x=table[300:].drop_columns("target"),
    )
    model.clear()
