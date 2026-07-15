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

# Fit + Predict forward pass via key/value caching for fast inference:
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
