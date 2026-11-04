# Installation

The `structured-data-models` package is available from Python 3.11 and PyTorch 2.7 onwards.
Install from PyPi:

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
