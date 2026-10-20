# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build the optional CUDA extension when explicitly requested."""

import os
from pathlib import Path
from typing import Any

from setuptools import setup

options: dict[str, Any] = {}
if os.environ.get("SDM_BUILD_CUDA_KERNELS") == "1":
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension

    include = Path(os.environ["CUTLASS_INCLUDE_DIR"]).resolve()
    if not (include / "cutlass/gemm/device/gemm.h").is_file():
        raise FileNotFoundError(
            "CUTLASS_INCLUDE_DIR must contain CUTLASS headers"
        )
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")
    options["ext_modules"] = [
        CUDAExtension(
            "sdm._kernels._linear_rmsnorm_cuda",
            sources=["sdm/_kernels/cuda/linear_rmsnorm.cu"],
            include_dirs=[str(include)],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "--expt-extended-lambda"],
            },
        )
    ]
    options["cmdclass"] = {
        "build_ext": BuildExtension.with_options(use_ninja=False)
    }

setup(**options)
