# Structured Data Models

Python package for structured data models.

## Examples

For a quick start, run [examples/tabiclv2.py](examples/tabiclv2.py) or open the
[TabICL quickstart notebook](examples/TabICL_demo.ipynb). More examples are
listed in [examples/README.md](examples/README.md).

```python
import torch
from sklearn.datasets import load_breast_cancer

from sdm import TableTensor, infer_stypes
from sdm.models import TabICLv2
from sdm.processing import Recipe

df = load_breast_cancer(as_frame=True).frame

# A lossless, fully tensorized representation of the raw data on GPU:
table = TableTensor.from_pandas(
    df=df,
    stypes=infer_stypes(df),
    device=device,
)

# Access to a variety of pre-trained structured data models:
model = TabICLv2(device=device)

# Unified and custom recipes for pre- and post-processing:
recipe = Recipe(
    features=[
        ShuffleColumns(),
        ImputeMissing(),
        StandardScale(),
        SigmaClip(threshold=4.0),

    ],
    target=[
        ShuffleClasses(),
    ],
)

# Common execution interface:
with torch.amp.autocast(device.type, torch.bfloat16, enabled=table.is_cuda):
    model(
        x_context=table[:300].drop_columns("target"),
        y_context=table[:300, "target"],
        x_query=table[300:].drop_columns("target"),
        recipe=recipe,
        num_estimators=8,
    )
```
