# TabFM

{py:class}`~sdm.models.TabFM` provides checkpoint-compatible classification
and scalar-regression inference through the common SDM model interface. It can
run with randomly initialized cores, or load an explicitly obtained local
TabFM v1.0.0 PyTorch checkpoint.

## Checkpoints

Released TabFM weights are governed by the TabFM Non-Commercial License v1.0.
SDM does not download or redistribute them. To use pretrained weights, obtain
them separately under their applicable terms and provide a local directory
with both task variants:

```text
/path/to/tabfm-checkpoint/
├── classification/
│   ├── config.json
│   └── model.safetensors
└── regression/
    ├── config.json
    └── model.safetensors
```

The loader also accepts `pytorch_model.bin` in either variant directory, but
prefers `model.safetensors` when both are present. Loading is strict and has no
network fallback.

## Fit and predict

Integer targets select classification; floating-point targets select scalar
regression. Calling {py:meth}`~sdm.models.TabFM.fit` records projected context
state, and later {py:meth}`~sdm.models.TabFM.predict` calls reuse it without
retaining the raw context features or targets.

```python
from pathlib import Path

import torch

from sdm.models import TabFM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TabFM(
    pretrained=True,
    checkpoint_path=Path("/path/to/tabfm-checkpoint"),
    device=device,
    dtype=torch.bfloat16,
)

context = torch.randn(1, 128, 12, device=device, dtype=torch.bfloat16)
labels = torch.randint(0, 10, (1, 128), device=device)
model.fit(context, labels)

first_query = torch.randn(1, 16, 12, device=device, dtype=torch.bfloat16)
second_query = torch.randn(1, 8, 12, device=device, dtype=torch.bfloat16)
first_logits = model.predict(first_query)
second_logits = model.predict(second_query)
```

For regression, pass floating-point context targets. The output has one value
per query row instead of classification logits. Use
{py:func}`~sdm.models.tabfm.default_regression_recipe` when the upstream-style
numeric preprocessing and target inverse transformation are needed. Fit all
learned recipe state on context rows only.

## Cache constraints

Cached prediction requires the query to preserve the fitted context's:

- leading batch shape and feature count;
- categorical routing mask and active feature widths (`d`), when supplied;
- model and input device; and
- model parameter dtype.

Different query-row counts are supported across repeated calls. The cache
stores projected state for both column stages and every dataset-wise ICL
layer. Row interaction is not cached: it uses rotary position embeddings and
processes each row independently. Cache-enabled encoders reject rotary
position embeddings so that replay cannot reuse position-dependent keys or
values incorrectly.

{py:meth}`~sdm.models.TabFM.clear` discards fitted context state.
