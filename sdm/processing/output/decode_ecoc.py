# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
from typing import cast

import torch
from torch import Tensor

from sdm import EnsembleTable, Stype, StypeLike, TableTensor
from sdm.processing import EnsembleProcessor
from sdm.processing.categorical.encode_ecoc import EncodeECOC


class DecodeECOC(EnsembleProcessor):
    """Map symbol scores back onto the original classes.

    The processor reads the codebook that :class:`EncodeECOC` fitted. For
    every member and every original class it selects the score of the symbol
    that the member gave that class, and it drops the positions where the
    member merged the class into the rest symbol. The input holds the logits
    over the symbols, so the processor normalizes them over the symbols first.
    The member dimension survives, so
    :class:`~sdm.processing.ReduceEstimators` averages the selected log scores
    and :class:`~sdm.processing.Softmax` normalizes them over the classes.

    Place this processor before :class:`~sdm.processing.ReduceEstimators`,
    which needs one score per original class.

    Args:
        encoder: The fitted target processor that holds the codebook.
    """

    handles_stypes = frozenset({Stype.numerical})
    requires_fit = False

    def __init__(self, encoder: EncodeECOC) -> None:
        super().__init__()
        # Held outside the module tree: the encoder already belongs to
        # `Recipe.target`, and `Recipe.output` must not require fitting.
        self._encoder = (encoder,)

    @property
    def encoder(self) -> EncodeECOC:
        """Return the target processor that holds the codebook."""
        return self._encoder[0]

    def _class_columns(self) -> dict[StypeLike, Sequence[str]]:
        return {
            Stype.numerical: [
                str(value) for value in self.encoder.categories.tolist()
            ]
        }

    def _scale(self, codebook: Tensor, dtype: torch.dtype) -> Tensor:
        # Rescaled by how many members separate each class, so that the mean
        # over members equals the sum over the members that carry the class.
        coverage = (codebook != self.encoder.rest).sum(dim=0).clamp_min(1)
        return (codebook.size(0) / coverage).to(dtype)

    def _symbol_index(self, table: TableTensor, codebook: Tensor) -> Tensor:
        # Symbols are read from the column names, not from their position, so
        # an upstream reordering of the alphabet cannot move a class.
        names = table.columns[Stype.numerical]
        position = torch.empty(len(names), dtype=torch.long)
        for index, name in enumerate(names):
            position[int(name)] = index
        return position.to(codebook.device)[codebook]

    def _check_members(self, found: int) -> int:
        expected = self.encoder.codebook.size(0)
        if found != expected:
            raise ValueError(
                f"'DecodeECOC' expects {expected} ensemble members "
                f"(got {found})"
            )
        return expected

    def _transform_ensemble(
        self,
        ensemble_table: EnsembleTable,
    ) -> EnsembleTable:
        if not self.encoder.active:
            return ensemble_table
        members = range(ensemble_table.num_members)
        tables = tuple(
            ensemble_table.table(member_id) for member_id in members
        )
        # Stacking aligns the members by column name, so a member that orders
        # the symbols differently still lands on the same axis.
        stacked = self._transform(cast(TableTensor, torch.stack(tables)))
        return EnsembleTable.from_tables(
            tables=[stacked[member_id] for member_id in members],
            member_table_ids=members,
        )

    def _transform(self, table: TableTensor) -> TableTensor:
        if not self.encoder.active:
            return table
        if table.dim() < 3:
            raise ValueError(
                "'DecodeECOC' expects a leading ensemble dimension"
            )

        scores = table.numerical
        num_members = self._check_members(scores.size(0))
        codebook = self.encoder.codebook.to(scores.device)
        num_classes = codebook.size(1)
        index = self._symbol_index(table, codebook)

        trailing = (1,) * (scores.dim() - 2)
        selected = scores.log_softmax(dim=-1).gather(
            -1,
            index.view(num_members, *trailing, num_classes).expand(
                *scores.shape[:-1],
                num_classes,
            ),
        )
        active = (codebook != self.encoder.rest).view(
            num_members,
            *trailing,
            num_classes,
        )
        return table.__class__(
            columns=self._class_columns(),
            numerical=torch.where(active, selected, selected.new_zeros(()))
            * self._scale(codebook, selected.dtype),
        )
