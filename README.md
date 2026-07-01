# Structured Data Models

Python package for structured data models.

```python
import torch
from sklearn.datasets import load_breast_cancer

from sdm import TableTensor
from sdm.models import TabICLv2
from sdm.processing import Recipe

df = load_breast_cancer(as_frame=True).frame

# A lossless, fully tensorized representation of the raw data on GPU:
table = TableTensor.from_pandas(df, device)

# Access to a variety of pre-trained structured data models:
model = TabICLv2(device=device)

# Unified and custom recipes for pre- and post-processing:
recipe = Recipe()

# Common execution interface:
with torch.amp.autocast(device.type, torch.bfloat16, enabled=table.is_cuda):
    model(
        x=table.drop_columns("target"),
        y=table[:300, "target"],
        recipe=recipe,
        num_estimators=8,
    )
```
