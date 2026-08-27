import torch
from sklearn.datasets import load_breast_cancer
from torch import Tensor

import sdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
df = load_breast_cancer(as_frame=True).frame

table = sdm.TableTensor.from_pandas(
    df=df,
    stypes=sdm.infer_stypes(df, overrides={"target": "categorical"}),
    device=device,
)
model = sdm.models.TabICLv2(device=device)

# Serving recipes (compile, bucketed padding): see benchmark_tabiclv2.py.

# Default in-context learning forward pass:
with torch.amp.autocast(device.type, torch.float16, enabled=table.is_cuda):
    model(
        x_context=table[:300].drop_columns("target"),
        y_context=table[:300, "target"],
        x_query=table[300:].drop_columns("target"),
        num_estimators=2,
    )

# Fit + Predict forward pass via key/value caching for fast inference:
with torch.amp.autocast(device.type, torch.float16, enabled=table.is_cuda):
    model.fit(
        x=table[:300].drop_columns("target"),
        y=table[:300, "target"],
        num_estimators=2,
    )
    model.predict(table[300:].drop_columns("target"))

model.clear()

# Capturing embeddings:
embeddings: list[Tensor] = []


def _embedding(module: torch.nn.Module, args: tuple[Tensor, ...]) -> None:
    embeddings.append(args[0])


head = model.models["classification"].icl_block.head
handle = head.register_forward_pre_hook(_embedding)
with torch.amp.autocast(device.type, torch.float16, enabled=table.is_cuda):
    model(
        x_context=table[:300].drop_columns("target"),
        y_context=table[:300, "target"],
        x_query=table[300:].drop_columns("target"),
    )
handle.remove()
