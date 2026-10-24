# Installation

The `structured-data-models` package supports Python 3.11 or later and PyTorch 2.7 or later.
Install from PyPI:

```bash
pip install structured-data-models
```

Alternatively, install from `main` via:

```bash
pip install git+https://github.com/NVIDIA/structured-data-models.git
```

```{note}
For CUDA workloads, we highly recommend installing [`cudf`](https://docs.rapids.ai/install) as an additional dependency to keep dataframe-style operations on GPU and avoid unnecessary data movement.
```
