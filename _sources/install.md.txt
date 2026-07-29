# Installation

The `structured-data-models` package is available from Python 3.10 and PyTorch 2.5 onwards.
Install via:

```bash
pip install structured-data-models
```

For CUDA workloads, we highly recommend installing [`cudf`](https://docs.rapids.ai/install) as an additional dependency.
Otherwise, some operations might fall back to a CPU backend, requiring device synchronization and unnecessary data movement.
