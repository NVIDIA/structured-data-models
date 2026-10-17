# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
from typing import cast

import torch
from torch import Tensor


class SeedMixin:
    r"""Store a seed at fit, from which transform draws its row values."""

    _seed = 0

    def get_extra_state(self) -> object:
        r""":meta private:"""  # noqa: D415
        return self._seed

    def set_extra_state(self, state: object) -> None:
        r""":meta private:"""  # noqa: D415
        self._seed = cast(int, state)

    def _draw_seed(self, generator: torch.Generator | None) -> None:
        device = None if generator is None else generator.device
        self._seed = int(
            torch.randint(2**62, (), device=device, generator=generator)
        )

    def _transform_generator(self) -> torch.Generator:
        # A CPU generator makes the output independent of the device.
        return torch.Generator().manual_seed(self._seed)


def uniform(
    low: float,
    high: float,
    size: Sequence[int],
    *,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator | None,
) -> Tensor:
    return torch.empty(size, device=device, dtype=dtype).uniform_(
        low, high, generator=generator
    )
