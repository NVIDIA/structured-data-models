# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import functools
from typing import Literal

import torch
from torch import Tensor

from sdm import (
    CategoricalTensor,
    EnsembleData,
    EnsembleTable,
    Stype,
    TableTensor,
)
from sdm.processing import Processor
from sdm.tensor.categorical import _iter_categories


class ShuffleCategories(Processor):
    """Permute the integer codes of categorical columns.

    One permutation per categorical column is drawn when the processor is
    fitted. Codes and their corresponding category vectors are permuted
    together so decoded values remain unchanged. Negative codes represent
    missing values and are preserved unchanged. Only categorical columns
    are supported; use :class:`~sdm.processing.StypeDispatch` to apply this
    processor to the categorical block of a mixed table.

    Args:
        method: Permutation strategy. ``"random"`` remaps codes
            with an independent permutation. ``"shift"`` uses a cyclic
            offset, cycling through every offset in random order.
    """

    handles_stypes = (Stype.categorical,)
    requires_fit = True

    def __init__(
        self,
        method: Literal["shift", "random"] = "random",
    ) -> None:
        super().__init__()
        self.method = method
        self.register_buffer("_permutations", None)

    def _draw_permutations(
        self,
        table: TableTensor,
        *,
        unused_offsets: dict[tuple[str, int], Tensor],
        generator: torch.Generator | None = None,
    ) -> list[Tensor]:
        device = table.categorical.device
        permutations: list[Tensor] = []
        for column_index, category in enumerate(table.categorical.categories):
            n_classes = category.numel()
            if n_classes <= 1:
                permutation = torch.arange(n_classes, device=device)
            elif self.method == "shift":
                column_name = table.categorical_columns[column_index]
                key = (column_name, n_classes)
                offsets = unused_offsets.get(key)
                if offsets is None or offsets.numel() == 0:
                    offsets = torch.randperm(
                        n_classes,
                        generator=generator,
                        device=device,
                    )
                offset = offsets[:1]
                unused_offsets[key] = offsets[1:]
                permutation = (
                    torch.arange(n_classes, device=device) - offset
                ) % n_classes
            else:
                assert self.method == "random"
                permutation = torch.randperm(
                    n_classes,
                    generator=generator,
                    device=device,
                )
            permutations.append(permutation)
        return permutations

    def _fit(
        self,
        table: TableTensor | EnsembleTable,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        unused_offsets: dict[tuple[str, int], Tensor] = {}
        draw = functools.partial(
            self._draw_state,
            unused_offsets=unused_offsets,
            generator=generator,
        )
        permutations = EnsembleData.execute(
            draw, table, randomness="different"
        )
        self.register_buffer("_permutations", permutations)

    def _draw_state(
        self,
        table: TableTensor,
        *,
        unused_offsets: dict[tuple[str, int], Tensor],
        generator: torch.Generator | None,
    ) -> CategoricalTensor:
        permutations = self._draw_permutations(
            table, unused_offsets=unused_offsets, generator=generator
        )
        return CategoricalTensor(
            table.categorical.code.new_empty((0, len(permutations))),
            categories=permutations,
        )

    def _transform(
        self, table: TableTensor | EnsembleTable
    ) -> TableTensor | EnsembleTable:
        return EnsembleData.execute(
            self._permute, table, self._permutations, batched=True
        )

    @staticmethod
    def _permute(
        table: TableTensor,
        state: CategoricalTensor,
    ) -> TableTensor:
        permutations = tuple(_iter_categories(state))
        code = table.categorical.code.clone()
        valid_mask = table.categorical.isfinite()
        categories: list[Tensor] = []
        for index, category in zip(
            range(len(permutations)),
            _iter_categories(table.categorical),
            strict=True,
        ):
            permutation = permutations[index]
            codes = code[..., index]
            valid = valid_mask[..., index]
            if permutation.numel() > 0:
                mapped = permutation[codes.clamp_min(0).long()]
                code[..., index] = torch.where(
                    valid, mapped.to(codes.dtype), codes
                )
            categories.append(category[permutation.argsort()])

        categorical = CategoricalTensor(
            code=code,
            categories=categories,
        )
        return table.replace_blocks(categorical=categorical)

    def __repr__(self, *, indent: int = 0) -> str:
        return (
            f"{' ' * indent}{self.__class__.__name__}(method={self.method!r})"
        )
