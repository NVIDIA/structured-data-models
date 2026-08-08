# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modified for the structured-data-models package.

import itertools
from typing import cast

import torch
from torch import Tensor

from sdm import CategoricalTensor, Stype, TableTensor
from sdm.processing import EnsembleProcessor
from sdm.tensor import EnsembleTable

_MAX_FEATURES = 500


class _FeaturePermutation(torch.nn.Module):
    """Store one fitted feature permutation."""

    indices: Tensor
    order: tuple[int, ...]

    def __init__(self, indices: Tensor) -> None:
        super().__init__()
        self.register_buffer("indices", indices, persistent=False)
        self.order = tuple(indices.tolist())


class _ShuffleFeatures(EnsembleProcessor):
    """Draw TabFM's vectorized ensemble feature permutations."""

    handles_stypes = frozenset({Stype.numerical})
    requires_fit = True

    def __init__(self) -> None:
        super().__init__()
        self._permutations = torch.nn.ModuleList()

    @staticmethod
    def _sample_ordered(
        num_features: int,
        *,
        device: torch.device,
        generator: torch.Generator | None,
    ) -> Tensor:
        """Sample 500 indices without full-width temporary storage."""
        selected = torch.empty(0, dtype=torch.long, device=device)
        while selected.numel() < _MAX_FEATURES:
            remaining = _MAX_FEATURES - selected.numel()
            candidates = torch.randint(
                num_features,
                (max(2 * remaining, _MAX_FEATURES),),
                generator=generator,
                device=device,
            )
            pool = torch.cat((selected, candidates))
            duplicate = (
                pool[:, None].eq(pool[None, :]).tril(diagonal=-1).any(dim=-1)
            )
            selected = pool[~duplicate][:_MAX_FEATURES]
        return selected.clone()

    def _draw_permutations(
        self,
        table: TableTensor,
        num_members: int,
        *,
        generator: torch.Generator | None,
    ) -> tuple[Tensor, ...]:
        num_features = table.numerical.size(-1)
        device = table.device
        if num_members == 1 and num_features <= _MAX_FEATURES:
            return (torch.arange(num_features, device=device),)

        draw_device = table.device if generator is None else generator.device
        if num_features <= 5:
            permutations = tuple(itertools.permutations(range(num_features)))
            ids = torch.randperm(
                len(permutations),
                generator=generator,
                device=draw_device,
            ).tolist()
            selected = tuple(permutations[index] for index in ids)
            repeated = tuple(
                selected[index % len(selected)] for index in range(num_members)
            )
            assignment = torch.randperm(
                num_members,
                generator=generator,
                device=draw_device,
            ).tolist()
            return tuple(
                torch.tensor(
                    repeated[index],
                    device=device,
                    dtype=torch.long,
                )
                for index in assignment
            )

        if num_features <= _MAX_FEATURES:
            return tuple(
                torch.randperm(
                    num_features,
                    generator=generator,
                    device=draw_device,
                ).to(device)
                for _ in range(num_members)
            )

        return tuple(
            self._sample_ordered(
                num_features=num_features,
                device=draw_device,
                generator=generator,
            ).to(device)
            for _ in range(num_members)
        )

    def _fit_ensemble(
        self,
        ensemble_table: EnsembleTable,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        permutations = self._draw_permutations(
            ensemble_table.table(0),
            ensemble_table.num_members,
            generator=generator,
        )
        self._permutations = torch.nn.ModuleList(
            _FeaturePermutation(permutation) for permutation in permutations
        )

    def _transform_ensemble(
        self,
        ensemble_table: EnsembleTable,
    ) -> EnsembleTable:
        if len(self._permutations) != ensemble_table.num_members:
            raise RuntimeError(
                "_ShuffleFeatures must be fitted with the same number of "
                "ensemble members before transform."
            )

        tables = []
        for member_id, permutation in enumerate(self._permutations):
            table = ensemble_table.table(member_id)
            permutation = cast(_FeaturePermutation, permutation)
            tables.append(
                table.__class__(
                    columns={
                        Stype.numerical: tuple(
                            table.columns[Stype.numerical][index]
                            for index in permutation.order
                        )
                    },
                    numerical=table.numerical.index_select(
                        -1,
                        permutation.indices,
                    ),
                )
            )
        return EnsembleTable.from_tables(
            tables=tables,
            member_table_ids=range(len(tables)),
        )


class _ShiftClasses(EnsembleProcessor):
    """Apply balanced class-ID shifts with a canonical first member."""

    handles_stypes = frozenset({Stype.categorical})
    requires_fit = True

    def __init__(self) -> None:
        super().__init__()
        self._offsets: tuple[int, ...] = ()

    def _fit_ensemble(
        self,
        ensemble_table: EnsembleTable,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        table = ensemble_table.table(0)
        num_classes = table.categorical.categories[0].numel()
        num_members = ensemble_table.num_members
        if num_members == 1 or num_classes <= 1:
            self._offsets = (0,) * num_members
            return

        draw_device = table.device if generator is None else generator.device
        cycle = torch.randperm(
            num_classes,
            generator=generator,
            device=draw_device,
        )
        offsets = cycle.repeat((num_members + num_classes - 1) // num_classes)[
            :num_members
        ]
        offsets = offsets[
            torch.randperm(
                num_members,
                generator=generator,
                device=draw_device,
            )
        ]
        zero_indices = offsets.eq(0).nonzero()
        if zero_indices.numel() == 0:
            offsets[0] = 0
        else:
            zero_index = zero_indices[0, 0]
            first = offsets[0].clone()
            offsets[0] = 0
            offsets[zero_index] = first
        self._offsets = tuple(offsets.tolist())

    def _transform_ensemble(
        self,
        ensemble_table: EnsembleTable,
    ) -> EnsembleTable:
        if len(self._offsets) != ensemble_table.num_members:
            raise RuntimeError(
                "_ShiftClasses must be fitted with the same number of "
                "ensemble members before transform."
            )

        tables = []
        for member_id, offset in enumerate(self._offsets):
            table = ensemble_table.table(member_id)
            categorical = table.categorical
            category = categorical.categories[0]
            code = categorical.code
            shifted = torch.where(
                code >= 0,
                (code + offset) % max(category.numel(), 1),
                code,
            )
            category_indices = (
                torch.arange(category.numel(), device=table.device) - offset
            ) % max(category.numel(), 1)
            tables.append(
                table.replace_blocks(
                    categorical=CategoricalTensor(
                        code=shifted,
                        categories=(category[category_indices],),
                    )
                )
            )
        return EnsembleTable.from_tables(
            tables=tables,
            member_table_ids=range(len(tables)),
        )
