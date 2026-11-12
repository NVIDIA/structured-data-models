# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Literal, cast

import torch
from torch import Tensor

from sdm import (
    EnsembleData,
    EnsembleTable,
    EnsembleTensor,
    Stype,
    TableLike,
    TableTensor,
)
from sdm.processing import InvertibleMixin, Processor


class ShuffleColumns(Processor, InvertibleMixin):
    """Permute numerical columns and their names.

    Args:
        method: Permutation strategy. ``"random"`` draws independent
            permutations, and ``"latin"`` draws coupled Latin
            permutations.
    """

    handles_stypes = (Stype.numerical,)
    requires_fit = True

    _permutations: EnsembleTensor[Tensor]

    def __init__(
        self,
        method: Literal["random", "latin"] = "random",
    ) -> None:
        super().__init__()
        self.method = method
        self.register_buffer("_permutations", None)
        self._host_permutations: tuple[tuple[int, ...], ...] = ()

    def get_extra_state(self) -> tuple[tuple[int, ...], ...]:
        r""":meta private:"""  # noqa: D415
        return self._host_permutations

    def set_extra_state(self, state: object) -> None:
        r""":meta private:"""  # noqa: D415
        self._host_permutations = cast(tuple[tuple[int, ...], ...], state)

    def _fit(
        self,
        table: TableLike,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        widths = EnsembleData.execute(Tensor.size, table.numerical, -1)
        max_features = (
            max(widths) if isinstance(widths, EnsembleData) else widths
        )
        base = rows = None
        if self.method == "latin":
            base = torch.randperm(
                max_features, generator=generator, device=table.device
            )
            rows = torch.randperm(
                max_features, generator=generator, device=table.device
            )
        members = (
            EnsembleData(
                groups=tuple(range(len(table))),
                locations=tuple((i, 0) for i in range(len(table))),
            )
            if isinstance(table, EnsembleTable)
            else 0
        )
        permutations = EnsembleData.execute(
            self._draw_permutation,
            table.numerical,
            members,
            base,
            rows,
            generator=generator,
            randomness="different",
        )
        self.register_buffer("_permutations", permutations)
        self._host_permutations = (
            tuple(tuple(row.tolist()) for row in permutations)
            if isinstance(permutations, EnsembleTensor)
            else (tuple(permutations.tolist()),)
        )

    def _draw_permutation(
        self,
        numerical: Tensor,
        member: int,
        base: Tensor | None,
        rows: Tensor | None,
        *,
        generator: torch.Generator | None,
    ) -> Tensor:
        width = numerical.size(-1)
        if width <= 1:
            return torch.arange(width, device=numerical.device)
        if self.method == "random":
            return torch.randperm(
                width, generator=generator, device=numerical.device
            )
        assert base is not None
        assert rows is not None
        if width < base.numel():
            base = base[base < width]
            rows = rows[rows < width]
        return base[(member % width - rows) % width]

    def _transform(self, table: TableLike) -> TableLike:
        return self._permute(table, inverse=False)

    def _inverse_transform(self, table: TableLike) -> TableLike:
        return self._permute(table, inverse=True)

    def _permute(self, table: TableLike, *, inverse: bool) -> TableLike:
        host_permutations = (
            EnsembleData(
                groups=self._host_permutations,
                locations=tuple(
                    (i, 0) for i in range(len(self._host_permutations))
                ),
            )
            if isinstance(table, EnsembleTable)
            else self._host_permutations[0]
        )
        permutation = (
            self._permutations.argsort(dim=-1)
            if inverse
            else self._permutations
        )
        indices = permutation.unsqueeze(-2).expand_as(table.numerical)
        numerical = table.numerical.gather(-1, indices)
        columns = EnsembleData.execute(
            self._column_names,
            table.numerical_columns,
            host_permutations,
            inverse=inverse,
        )
        shuffled = EnsembleData.execute(
            TableTensor,
            columns={Stype.numerical: columns},
            numerical=numerical,
        )
        return cast(
            TableLike,
            EnsembleData.execute(
                torch.cat,
                (table.drop_stypes(Stype.numerical), shuffled),
                dim=-1,
            ),
        )

    @staticmethod
    def _column_names(
        names: tuple[str, ...], permutation: tuple[int, ...], *, inverse: bool
    ) -> tuple[str, ...]:
        if inverse:
            indices = [0] * len(permutation)
            for destination, source in enumerate(permutation):
                indices[source] = destination
            permutation = tuple(indices)
        return tuple(names[index] for index in permutation)

    def __repr__(self, *, indent: int = 0) -> str:
        return (
            f"{' ' * indent}{self.__class__.__name__}(method={self.method!r})"
        )
