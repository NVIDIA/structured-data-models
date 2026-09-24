# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Collection
from typing import cast

import torch

from sdm import Stype, TableTensor
from sdm.processing import Processor


class PCA(Processor):
    r"""Project numerical columns onto their principal components.

    Args:
        num_components: Number of principal components to keep.
        append_original: Keep the original numerical columns and append the
            components to them.
        center: Subtract the column mean before the projection. Without it
            the projection is a truncated singular value decomposition.
        exclude_columns: Columns that do not take part in the projection.
            Requires :obj:`append_original`, which keeps those columns.
    """

    handles_stypes = frozenset({Stype.numerical})
    requires_fit = True

    def __init__(
        self,
        num_components: int,
        *,
        append_original: bool = False,
        center: bool = True,
        exclude_columns: Collection[str] = (),
    ) -> None:
        super().__init__()
        if num_components < 1:
            raise ValueError("'num_components' must be positive")
        if exclude_columns and not append_original:
            raise ValueError("'exclude_columns' requires 'append_original'")
        self.num_components = num_components
        self.append_original = append_original
        self.center = center
        self.exclude_columns = frozenset(exclude_columns)
        self.register_buffer("mean", torch.empty(0))
        self.register_buffer("components", torch.empty(0))

    def _projected(self, table: TableTensor) -> torch.Tensor:
        r"""Return the columns that take part in the projection."""
        excluded = self.exclude_columns & table.column_names
        numerical = table.drop_columns(excluded).numerical
        return numerical.masked_fill(numerical.isinf(), torch.nan)

    def _prepared(self, table: TableTensor) -> torch.Tensor:
        r"""Return the input of the decomposition.

        The method subtracts the mean, which is zero without centering, and
        replaces the missing values.
        """
        return (self._projected(table) - self.mean).nan_to_num(nan=0.0)

    def _fit(
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> None:

        if table.numerical.size(-1) == 0:
            raise ValueError(
                f"{self.__class__.__name__!r} requires 'table' to have "
                f"numerical features"
            )

        projected = self._projected(table)
        mean = projected.nanmean(dim=-2, keepdim=True)
        self.mean = mean if self.center else torch.zeros_like(mean)
        _, _, vh = torch.linalg.svd(self._prepared(table), full_matrices=False)
        num_components = min(
            self.num_components,
            projected.size(-2),
            projected.size(-1),
        )
        self.components = vh[..., :num_components, :].transpose(-2, -1)

    def _transform(self, table: TableTensor) -> TableTensor:
        x = self._prepared(table) @ self.components
        out = TableTensor(
            columns={Stype.numerical: [f"pca_{i}" for i in range(x.size(-1))]},
            numerical=x,
        )
        base = table
        if not self.append_original:
            base = table.drop_stypes(Stype.numerical)
        return cast(TableTensor, torch.cat([base, out], dim=-1))
