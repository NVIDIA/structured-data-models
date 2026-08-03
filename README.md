# Structured Data Models

Python package for structured data models.

## Installation

The `structured-data-models` package is available from Python 3.10 and PyTorch 2.5 onwards.
Install via:

```bash
pip install structured-data-models
```

> [!NOTE]
> For CUDA workloads, we highly recommend installing [`cudf`](https://docs.rapids.ai/install) as an additional dependency to keep dataframe-style operations on GPU and avoid unnecessary data movement.

## Quick Tour

```python
from sklearn.datasets import load_breast_cancer

import sdm

df = load_breast_cancer(as_frame=True).frame

# A lossless, fully tensorized representation of the raw data on GPU:
table = sdm.TableTensor.from_pandas(
    df=df,
    stypes=sdm.infer_stypes(df),
    device="cuda",
)

# Access to a variety of pre-trained structured data models:
model = sdm.models.TabICLv2(device="cuda")

# Default in-context learning forward pass:
model(
    x_context=table[:300].drop_columns("target"),
    y_context=table[:300, "target"],
    x_query=table[300:].drop_columns("target"),
    num_estimators=8,
)

# Fit + Predict forward pass via key/value caching for fast inference:
model.fit(
    x=table[:300].drop_columns("target"),
    y=table[:300, "target"],
    num_estimators=8,
)
model.predict(table[300:].drop_columns("target"))
model.clear()
```
