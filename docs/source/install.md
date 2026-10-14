# Installation

The `structured-data-models` package is available from Python 3.11 and PyTorch 2.7 onwards.
Choose a Python version supported by your PyTorch release.
Install from the `main` branch:

```bash
pip install git+https://github.com/NVIDIA/structured-data-models.git
```

```{note}
For CUDA workloads, we highly recommend installing [`cudf`](https://docs.rapids.ai/install) as an additional dependency to keep dataframe-style operations on GPU and avoid unnecessary data movement.
```

## Compilation compatibility

`torch.compile` has stricter Python compatibility than ordinary PyTorch execution.
PyTorch 2.14.0 and 2.14.1 support Python 3.15 for ordinary execution, but reject `torch.compile` on that Python version.
Python 3.15 compilation support is included in the PyTorch 2.15 release candidates ([upstream tracker](https://github.com/pytorch/pytorch/issues/184352)).
To compile with an older PyTorch release, use a Python version supported by that release's compiler.
Compiler availability does not guarantee that every SDM model or configuration compiles.
