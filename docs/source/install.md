# Installation

The `structured-data-models` package is available from Python 3.11 and PyTorch 2.7 onwards.
Install via:

```bash
pip install structured-data-models
```

```{note}
For CUDA workloads, we highly recommend installing [`cudf`](https://docs.rapids.ai/install) as an additional dependency to keep dataframe-style operations on GPU and avoid unnecessary data movement.
```

The optional `cudnn` extra (`pip install "structured-data-models[cudnn]"`) adds cuDNN variable-length attention for padded inputs, enabled via `sdm.nn.enable_cudnn_varlen()` (Linux only; on other platforms the extra installs nothing and the boolean-mask path is used).
