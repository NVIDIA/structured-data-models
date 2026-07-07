import torch
from sdm import Stype, TableTensor, infer_stypes
from sdm.models import TabICLv2
from sklearn.datasets import load_breast_cancer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df = load_breast_cancer(as_frame=True).frame

# `infer_stypes` returns a plain, mutable dict -- inspect it and correct any
# entry that doesn't match intent. Here, `target` is 0/1-encoded, so it
# infers as numerical by default even though it's really a categorical label.
stypes = infer_stypes(df)
stypes["target"] = Stype.categorical

table = TableTensor.from_pandas(df=df, stypes=stypes, device=device)

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
