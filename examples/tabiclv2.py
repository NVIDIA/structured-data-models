import torch
from sdm import TableTensor
from sdm.models import TabICLv2
from sklearn.datasets import load_breast_cancer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df = load_breast_cancer(as_frame=True).frame
table = TableTensor.from_pandas(
    df=df,
    stypes={
        **dict.fromkeys(df.columns[:-1], "numerical"),
        "target": "categorical",
    },
    device=device,
)

model = TabICLv2(device=device)

# Default in-context learning forward pass:
with torch.amp.autocast(device.type, torch.bfloat16, enabled=table.is_cuda):
    model(
        x=table.drop_columns("target"),
        y=table[:300, "target"],
    )

# Fit + Predict forward pass via key/value caching for fast inference:
with torch.amp.autocast(device.type, torch.bfloat16, enabled=table.is_cuda):
    model.fit(
        x=table[:300].drop_columns("target"),
        y=table[:300, "target"],
    )
    model.predict(
        x=table[300:].drop_columns("target"),
    )
    model.clear()
