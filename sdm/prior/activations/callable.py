# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor

from sdm.prior.activations import Activation


@dataclass(frozen=True)
class CallableActivation(Activation):
    """Adapt a callable to the :class:`Activation` interface.

    Args:
        function: Function that keeps the shape of its input.
    """

    function: Callable[[Tensor], Tensor]

    def forward(  # noqa: D102
        self, x: Tensor, generator: torch.Generator | None = None
    ) -> Tensor:
        return self.function(x)
