# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import cast

import torch
from torch import Tensor

from sdm import CategoricalTensor, Stype, TableTensor
from sdm.processing import Processor


class MaskRandom(Processor):
    """Mask numerical and categorical cells completely at random.

    Fit initializes a random stream. Each transform draws a missing-cell
    probability per table and masks each cell independently with that
    probability. Saving and loading preserves the next draw in the stream.

    Args:
        rate: Lower and upper bounds for uniform sampling of each table's
            missing-cell probability.
    """

    handles_stypes = (Stype.numerical, Stype.categorical)
    requires_fit = True

    def __init__(self, *, rate: tuple[float, float] = (0.01, 0.08)) -> None:
        super().__init__()
        self.rate = rate
        self._generator = torch.Generator()

    def get_extra_state(self) -> Tensor:
        r""":meta private:"""  # noqa: D415
        return self._generator.get_state()

    def set_extra_state(self, state: object) -> None:
        r""":meta private:"""  # noqa: D415
        self._generator.set_state(cast(Tensor, state).cpu())

    def _fit(
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        device = None if generator is None else generator.device
        seed = int(
            torch.randint(2**62, (), device=device, generator=generator)
        )
        self._generator.manual_seed(seed)

    def _transform(self, table: TableTensor) -> TableTensor:
        # Keep the random stream portable across CPU and CUDA inputs.
        missing_rate = torch.empty(
            size=table.size()[:-2],
            device="cpu",
        ).uniform_(*self.rate, generator=self._generator)
        num_numerical = table.numerical.size(-1)
        num_columns = num_numerical + table.categorical.size(-1)
        u = torch.rand(
            (*table.size()[:-1], num_columns),
            device="cpu",
            generator=self._generator,
        )
        missing = (u < missing_rate[..., None, None]).to(table.device)
        return table.replace_blocks(
            numerical=table.numerical.masked_fill(
                missing[..., :num_numerical], torch.nan
            ),
            categorical=CategoricalTensor(
                code=table.categorical.code.masked_fill(
                    missing[..., num_numerical:], -1
                ),
                categories=table.categorical.categories,
            ),
        )

    def __repr__(self, *, indent: int = 0) -> str:
        return f"{' ' * indent}{self.__class__.__name__}(rate={self.rate})"
