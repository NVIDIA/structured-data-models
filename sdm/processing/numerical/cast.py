# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from sdm import Stype, TableTensor
from sdm.processing import Processor
from sdm.processing.numerical._stats import _high_precision_dtype


class Cast(Processor):
    """Cast numerical columns to a floating-point dtype.

    On MPS devices, which do not support ``torch.float64``, casting to
    ``torch.float64`` falls back to ``torch.float32``.

    Args:
        dtype: The floating-point dtype of the numerical columns.
    """

    handles_stypes = frozenset({Stype.numerical})
    requires_fit = False

    def __init__(self, dtype: torch.dtype) -> None:
        super().__init__()
        if not dtype.is_floating_point:
            raise ValueError(f"Expected a floating-point dtype (got {dtype})")
        self.dtype = dtype

    def _transform(self, table: TableTensor) -> TableTensor:
        dtype = self.dtype
        if dtype == torch.float64:
            dtype = _high_precision_dtype(table.numerical.device)
        return table.replace_blocks(numerical=table.numerical.to(dtype))

    def __repr__(self, *, indent: int = 0) -> str:
        return f"{' ' * indent}{self.__class__.__name__}({self.dtype})"
