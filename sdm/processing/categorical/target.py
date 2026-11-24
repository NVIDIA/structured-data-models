# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import cast

import torch
from torch import Tensor
from torch.nn import ModuleList

from sdm import EnsembleTable, Stype, TableTensor
from sdm.nn._buffer import BufferList
from sdm.processing import TargetAwareEnsembleProcessor
from sdm.processing.categorical.align import AlignCategories
from sdm.tensor.categorical import _iter_categories


class AddTargetEncoding(TargetAwareEnsembleProcessor):
    r"""Add smoothed target statistics for categorical columns.

    Use in :class:`~sdm.processing.Recipe` features to access context labels.
    For a category with count :math:`n` and mean target :math:`\mu_c`, the
    encoding is :math:`(n\mu_c + s\mu)/(n+s)`, where :math:`\mu` is the global
    target mean and :math:`s` is ``smooth``. Classification encodes class
    probabilities, retaining only the positive class for binary targets and
    every class for multiclass targets, in sorted label order.

    ``fit_transform`` uses shuffled folds to exclude each row's fold from
    its statistics. Subsequent ``transform`` calls use all context labels.
    Missing values form a category; unseen values use the global mean.
    One target column is supported. Added columns are named
    ``<column>__target`` or ``<column>__target_<class_index>`` for multiclass.

    Args:
        smooth: Nonnegative smoothing strength. Zero uses raw category means.
        num_folds: Number of cross-fitting folds, capped at the context row
            count. Cross-fitting requires at least two folds and two rows.
    """

    handles_stypes = (Stype.categorical,)
    requires_fit = True

    def __init__(self, smooth: float = 0.0, num_folds: int = 5) -> None:
        super().__init__()
        self.smooth = smooth
        self.num_folds = num_folds
        self._aligners: ModuleList[AlignCategories] = ModuleList()
        self._encodings: BufferList[Tensor] = BufferList()
        self._columns: tuple[tuple[str, ...], ...] = ()
        self._member_ids: tuple[int, ...] = ()

    def get_extra_state(
        self,
    ) -> tuple[tuple[tuple[str, ...], ...], tuple[int, ...]]:
        r""":meta private:"""  # noqa: D415
        return self._columns, self._member_ids

    def set_extra_state(self, state: object) -> None:
        r""":meta private:"""  # noqa: D415
        self._columns, self._member_ids = cast(
            tuple[tuple[tuple[str, ...], ...], tuple[int, ...]], state
        )
        self._aligners = ModuleList([AlignCategories() for _ in self._columns])

    def _fit_ensemble(
        self,
        ensemble_table: EnsembleTable,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        self._fit_encode(ensemble_table, crossfit=False, generator=generator)

    def _fit_transform_ensemble(
        self,
        ensemble_table: EnsembleTable,
        *,
        generator: torch.Generator | None = None,
    ) -> EnsembleTable:
        return self._fit_encode(
            ensemble_table, crossfit=True, generator=generator
        )

    def _fit_encode(
        self,
        ensemble_table: EnsembleTable,
        *,
        crossfit: bool,
        generator: torch.Generator | None,
    ) -> EnsembleTable:
        if self._target is None:
            raise RuntimeError(
                "'AddTargetEncoding' requires context targets; use it in "
                "a 'Recipe' through model execution"
            )
        if len(ensemble_table) != len(self._target):
            raise ValueError("Expected one target table per feature table")

        self._aligners = ModuleList()
        self._encodings = BufferList()
        columns, outputs, member_ids, encodings = [], [], [], []
        state_ids: dict[tuple[tuple[int, int], tuple[int, int]], int] = {}
        for member_id, locations in enumerate(
            zip(ensemble_table._locations, self._target._locations)
        ):
            if locations in state_ids:
                member_ids.append(state_ids[locations])
                continue
            state_ids[locations] = len(columns)
            member_ids.append(len(columns))
            table, target = ensemble_table[member_id], self._target[member_id]
            if target.size(-1) != 1:
                raise ValueError("Expected exactly one target column")

            aligner = AlignCategories()
            categorical = table.select_stypes(Stype.categorical)
            if categorical.size(-1) == 0:
                self._aligners.append(aligner)
                encodings.append(table.numerical.new_empty(0))
                columns.append(())
                if crossfit:
                    outputs.append(table)
                continue
            flat = aligner.fit_transform(
                cast(
                    TableTensor,
                    categorical.contiguous().view(-1, categorical.size(-1)),
                )
            )
            codes = flat.categorical.code.reshape(categorical.size()).long()
            self._aligners.append(aligner)
            num_slots = (
                max(
                    (c.numel() for c in _iter_categories(flat.categorical)),
                    default=0,
                )
                + 2
            )  # Separate missing-value and unseen-value slots.
            codes = codes.masked_fill(codes < 0, num_slots - 2)

            dtype = (
                torch.float64
                if table.dtype == torch.float64
                else torch.float32
            )
            if target.categorical.size(-1):
                labels = (
                    AlignCategories(sort_by="value")
                    .fit_transform(
                        cast(TableTensor, target.contiguous().view(-1, 1))
                    )
                    .categorical
                )
                num_classes = labels.categories[0].numel()
                values = torch.nn.functional.one_hot(
                    labels.code.long().reshape(target.size()[:-1]),
                    num_classes=num_classes,
                ).to(dtype)
                if num_classes == 2:
                    values = values[..., 1:]
            else:
                values = target.numerical.to(dtype)

            width = values.size(-1)
            names = tuple(
                f"{column}__target" + (f"_{i}" if width > 1 else "")
                for column in table.categorical_columns
                for i in range(width)
            )
            columns.append(names)
            sums, counts = _statistics(codes, values, num_slots)
            prior = values.mean(dim=-2)
            encodings.append(_smooth(sums, counts, prior, self.smooth))
            if not crossfit:
                continue

            num_rows = table.size(-2)
            num_folds = min(self.num_folds, num_rows)
            if num_folds < 2:
                raise ValueError("Cross-fitting requires at least two folds")
            perm = torch.randperm(
                num_rows, device=table.device, generator=generator
            )
            encoded = values.new_empty((*codes.shape, width))
            start = 0
            for held in perm.tensor_split(num_folds):
                end = start + held.numel()
                train = torch.cat([perm[:start], perm[end:]])
                train_values = values[..., train, :]
                train_sums, train_counts = _statistics(
                    codes[..., train, :], train_values, num_slots
                )
                lookup = _smooth(
                    train_sums,
                    train_counts,
                    train_values.mean(dim=-2),
                    self.smooth,
                )
                encoded[..., held, :, :] = _lookup(lookup, codes[..., held, :])
                start = end
            outputs.append(_append(table, encoded, names))

        self._columns = tuple(columns)
        self._member_ids = tuple(member_ids)
        self._encodings = BufferList(encodings)
        if not crossfit:
            return ensemble_table
        return ensemble_table.replace_tables(
            tables=outputs, member_table_ids=member_ids
        )

    def _transform_ensemble(
        self,
        ensemble_table: EnsembleTable,
    ) -> EnsembleTable:
        outputs, member_ids = [], []
        output_ids: dict[tuple[tuple[int, int], int], int] = {}
        for member_id, location in enumerate(ensemble_table._locations):
            state_id = self._member_ids[member_id]
            key = (location, state_id)
            if key not in output_ids:
                output_ids[key] = len(outputs)
                table = ensemble_table[member_id]
                if not self._columns[state_id]:
                    outputs.append(table)
                    member_ids.append(output_ids[key])
                    continue
                categorical = table.select_stypes(Stype.categorical)
                flat = self._aligners[state_id].transform(
                    cast(
                        TableTensor,
                        categorical.contiguous().view(
                            -1, categorical.size(-1)
                        ),
                    )
                )
                codes = flat.categorical.code.reshape(
                    categorical.size()
                ).long()
                lookup = self._encodings[state_id]
                num_slots = lookup.size(-2)
                codes = codes.masked_fill(codes < 0, num_slots - 1)
                codes = codes.masked_fill(
                    categorical.categorical.code < 0, num_slots - 2
                )
                outputs.append(
                    _append(
                        table, _lookup(lookup, codes), self._columns[state_id]
                    )
                )
            member_ids.append(output_ids[key])
        return ensemble_table.replace_tables(
            tables=outputs, member_table_ids=member_ids
        )

    def __repr__(self, *, indent: int = 0) -> str:
        return (
            f"{' ' * indent}{self.__class__.__name__}("
            f"smooth={self.smooth!r}, num_folds={self.num_folds!r})"
        )


def _statistics(
    codes: Tensor, values: Tensor, num_slots: int
) -> tuple[Tensor, Tensor]:
    # Flatten column/category pairs so one scatter reduces all columns.
    num_columns, width = codes.size(-1), values.size(-1)
    offsets = torch.arange(num_columns, device=codes.device) * num_slots
    index = (codes + offsets).flatten(-2).unsqueeze(-1)
    sums = values.new_zeros(
        (*codes.shape[:-2], num_columns * num_slots, width)
    )
    counts = values.new_zeros((*sums.shape[:-1], 1))
    source = values.unsqueeze(-2).expand(*codes.shape, width).flatten(-3, -2)
    sums.scatter_add_(-2, index.expand_as(source), source)
    counts.scatter_add_(-2, index, torch.ones_like(index, dtype=values.dtype))
    return (
        sums.unflatten(-2, (num_columns, num_slots)),
        counts.unflatten(-2, (num_columns, num_slots)),
    )


def _smooth(
    sums: Tensor, counts: Tensor, prior: Tensor, smooth: float
) -> Tensor:
    prior = prior.unsqueeze(-2).unsqueeze(-2)
    return torch.where(
        counts > 0,
        (sums + smooth * prior) / (counts + smooth).clamp_min(1),
        prior,
    )


def _lookup(lookup: Tensor, codes: Tensor) -> Tensor:
    index = codes.transpose(-2, -1).unsqueeze(-1)
    index = index.expand(*index.shape[:-1], lookup.size(-1))
    return lookup.gather(-2, index).transpose(-3, -2)


def _append(
    table: TableTensor, values: Tensor, columns: tuple[str, ...]
) -> TableTensor:
    encoded = TableTensor(
        columns={Stype.numerical: columns},
        numerical=values.flatten(-2).to(table.dtype),
    )
    return cast(TableTensor, torch.cat([table, encoded], dim=-1))
