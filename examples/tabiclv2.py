import torch
from sdm import TableTensor, infer_stypes
from sdm.models import TabICLv2
from sklearn.datasets import load_diabetes

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
df = load_diabetes(as_frame=True).frame

table = TableTensor.from_pandas(
    df=df,
    stypes=infer_stypes(df),
    device=device,
)
model = TabICLv2(device=device)

# Default in-context learning forward pass:
with torch.amp.autocast(device.type, torch.bfloat16, enabled=table.is_cuda):
    model(
        x=table.drop_columns("target"),
        y=table[:300, "target"],
        recipe=model.default_recipe(),
        num_estimators=2,
    )

# Fit + Predict forward pass via key/value caching for fast inference. The
# recipe is fitted on the in-context examples and reused by predict calls:
with torch.amp.autocast(device.type, torch.bfloat16, enabled=table.is_cuda):
    model.fit(
        x=table[:300].drop_columns("target"),
        y=table[:300, "target"],
        recipe=model.default_recipe(),
        num_estimators=2,
    )
    model.predict(
        x=table[300:].drop_columns("target"),
    )
    model.clear()
