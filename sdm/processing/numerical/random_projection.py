# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Literal, cast

import torch

from sdm import EnsembleData, EnsembleTable, EnsembleTensor, Stype, TableTensor
from sdm.processing import Processor


class RandomProjection(Processor):
    r"""Randomly project numerical columns.

    Args:
        channels: Number of output numerical columns.
        init: The projection weight initialization.
            ``"normal"`` samples Gaussian weights scaled by
            ``1 / sqrt(channels)``.
    """

    handles_stypes = (Stype.numerical,)
    requires_fit = True

    _weights: EnsembleTensor[torch.Tensor]

    def __init__(
        self,
        channels: int,
        *,
        init: Literal["normal"] = "normal",
    ) -> None:
        super().__init__()
        self.channels = channels
        self.init = init
        self.register_buffer("_weights", None)

    def _fit(
        self,
        table: TableTensor | EnsembleTable,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        weights = EnsembleData.execute(
            self._draw_weights,
            table.numerical,
            generator=generator,
            randomness="different",
        )
        self.register_buffer("_weights", weights)

    def _draw_weights(
        self, numerical: torch.Tensor, *, generator: torch.Generator | None
    ) -> torch.Tensor:
        weights = numerical.new_empty(
            (*numerical.shape[:-2], self.channels, numerical.size(-1))
        )
        assert self.init == "normal"
        return weights.normal_(std=self.channels**-0.5, generator=generator)

    def _transform(
        self, table: TableTensor | EnsembleTable
    ) -> TableTensor | EnsembleTable:
        return EnsembleData.execute(
            self._project, table, self._weights, batched=True
        )

    def _project(
        self, table: TableTensor, weights: torch.Tensor
    ) -> TableTensor:
        if table.numerical.size(-1) == 0:
            return table
        projected = TableTensor(
            columns={
                Stype.numerical: tuple(f"rp_{i}" for i in range(self.channels))
            },
            numerical=table.numerical @ weights.transpose(-1, -2),
        )
        return cast(
            TableTensor,
            torch.cat((table.drop_stypes(Stype.numerical), projected), dim=-1),
        )

    def __repr__(self, *, indent: int = 0) -> str:
        return (
            f"{' ' * indent}{self.__class__.__name__}("
            f"channels={self.channels!r})"
        )
