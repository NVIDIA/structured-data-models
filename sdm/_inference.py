# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import contextlib
from typing import Literal

import torch


def inference_mode(
    mode: Literal["inference", "no_grad", "grad", "none"] = "inference",
) -> contextlib.AbstractContextManager[None]:
    r"""Context manager to adjust PyTorch inference and autograd states.

    Args:
        mode: The desired mode. ``"inference"`` runs under
            :func:`torch.inference_mode` (falling back to :func:`torch.no_grad`
            inside compiled regions), ``"no_grad"`` runs under
            :func:`torch.no_grad`, ``"grad"`` runs under
            :func:`torch.enable_grad`, and ``"none"`` preserves the ambient
            PyTorch context.
    """
    if mode == "inference":
        # `torch.inference_mode` is not supported inside a compiled region:
        # https://github.com/pytorch/pytorch/issues/180823
        return (
            torch.no_grad()
            if torch.compiler.is_compiling()
            else torch.inference_mode()
        )
    elif mode == "no_grad":
        return torch.no_grad()
    elif mode == "grad":
        return torch.enable_grad()
    else:
        assert mode == "none"
        return contextlib.nullcontext()
