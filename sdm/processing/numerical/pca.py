# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import cast

import torch
from torch import Tensor

from sdm import EnsembleData, Stype, TableLike, TableTensor
from sdm.processing import Processor


class PCA(Processor):
    r"""Project numerical columns onto their principal components.

    Args:
        num_components: Maximum number of principal components, capped by the
            fitted row count. Tables with no more than this many numerical
            columns are returned unchanged.
    """

    handles_stypes = (Stype.numerical,)
    requires_fit = True

    mean: Tensor
    components: Tensor

    def __init__(self, num_components: int) -> None:
        super().__init__()
        self.num_components = num_components
        self.register_buffer("mean", torch.empty(0))
        self.register_buffer("components", torch.empty(0))

    def _fit(
        self,
        table: TableLike,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        mean, components = EnsembleData.execute(
            self._components, table.numerical, batched=True
        )
        self.register_buffer("mean", mean)
        self.register_buffer("components", components)

    def _components(self, numerical: Tensor) -> tuple[Tensor, Tensor]:

        if numerical.size(-1) <= self.num_components:
            empty = numerical.new_empty((*numerical.shape[:-2], 0))
            return empty, empty

        mean = numerical.mean(dim=-2, keepdim=True)
        _, _, vh = torch.linalg.svd(
            numerical - mean,
            full_matrices=False,
        )
        num_components = min(self.num_components, numerical.size(-2))
        return mean, vh[..., :num_components, :].transpose(-2, -1)

    def _transform(self, table: TableLike) -> TableLike:
        return EnsembleData.execute(
            self._project, table, self.mean, self.components, batched=True
        )

    def _project(
        self, table: TableTensor, mean: Tensor, components: Tensor
    ) -> TableTensor:
        if table.numerical.size(-1) <= self.num_components:
            return table

        x = (table.numerical - mean) @ components
        out = TableTensor(
            columns={Stype.numerical: [f"pca_{i}" for i in range(x.size(-1))]},
            numerical=x,
        )
        return cast(
            TableTensor,
            torch.cat([table.drop_stypes(Stype.numerical), out], dim=-1),
        )

    def __repr__(self, *, indent: int = 0) -> str:
        return (
            f"{' ' * indent}{self.__class__.__name__}("
            f"num_components={self.num_components})"
        )
