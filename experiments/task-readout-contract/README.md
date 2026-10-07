# Task readout operator contract

This branch tightens the runtime checks for the compiled TaskGraph readout
operator. Its fake implementation promises a one-dimensional tensor with one
entry per task row. The implementation previously accepted a readout tensor
with another shape.

```python
import torch
from sdm.models.kumo.relational.task import _validated_readout

torch.library.opcheck(
    _validated_readout,
    (torch.arange(3), torch.arange(4), 3, "Invalid readout"),
    test_utils=("test_faketensor",),
)
```

Before the change, the concrete result has shape `[4]`, while the fake result
has shape `[3]`; `opcheck` reports mismatched tensor metadata. The operator now
rejects this input with the existing `ValueError`. It also rejects a
multidimensional readout tensor. Valid task inputs are unchanged.

This is an unchecked custom-operator assumption, **not a demonstrated public
TaskGraph regression**: the current caller obtains equally sized task and
readout indices from the same join.

Validation on CPU, PyTorch 2.7.1 and 2.14.0:

| Check | Result on both versions |
|---|---|
| Focused shape regression tests | 2 passed |
| Empty/nonempty valid inputs, schema and fake checks | Passed |
| Actual Inductor, `fullgraph=False` and `fullgraph=True` | Passed |
| Invalid shapes under full-graph Inductor | Existing `ValueError` preserved |

Run the committed regression with:

```bash
python -m pytest test/models/kumo/relational/test_model.py -q -k validated_readout
```
