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

## Optional CUDA projection kernels

The default installation uses PyTorch and the existing SDM kernels. An optional native extension combines attention projection with per-head normalization and full split-half rotary embeddings. It supports inference with FP16 CUDA autocast on compute capability 12.0, using the cuBLAS backend. Other devices, precisions, training, customized transforms, and unsupported tensor layouts retain the existing execution path.

Build from a source checkout using the PyTorch environment that will run SDM. Install a compatible CUDA toolkit and C++ compiler, and provide the CUTLASS headers. The build has been exercised with PyTorch 2.14.1 (CUDA 13.0), CUDA toolkit 13.2, and CUTLASS revision `da5e086dab31d63815acafdac9a9c5893b1c69e2`.

```bash
python -m pip install 'setuptools>=80' 'setuptools-scm>=10' wheel
git clone https://github.com/NVIDIA/cutlass.git /path/to/cutlass
git -C /path/to/cutlass checkout da5e086dab31d63815acafdac9a9c5893b1c69e2
SDM_BUILD_CUDA_KERNELS=1 \
CUTLASS_INCLUDE_DIR=/path/to/cutlass/include \
TORCH_CUDA_ARCH_LIST=12.0 \
python -m pip install --no-build-isolation .
```

This opt-in build uses the installed PyTorch headers and libraries. Rebuild the extension after changing the PyTorch or CUDA environment. SDM does not download CUTLASS or compile this native extension during prediction.
