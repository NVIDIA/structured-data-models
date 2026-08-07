from typing import Literal

import torch
from torch import Tensor

from sdm import (
    CategoricalTensor,
    ColumnarTensor,
    StringTensor,
    Stype,
    TableTensor,
)
from sdm.processing import EnsembleProcessor
from sdm.relational.join import join_index
from sdm.tensor import EnsembleTable

_UNSIGNED_DTYPES = frozenset({torch.uint16, torch.uint32, torch.uint64})


class AlignCategories(EnsembleProcessor):
    """Align categorical columns to categories observed during fitting.

    A categorical column stores each value as an integer code indexing an
    ordered list of categories. Separately created tables can use different
    codes for the same value. Fitting learns the category list for each column,
    and transforming remaps another table to use it. Missing values and
    categories not seen during fitting receive code ``-1``.

    Args:
        sort_by: How to order fitted categories.
            ``"code"`` keeps observed categories in original order.
            ``"frequency"`` orders observed categories by descending frequency.
            ``"value"`` orders observed categories by ascending value.

    >>> import pandas as pd
    >>> import sdm
    >>> from sdm.processing import AlignCategories
    >>> table1 = sdm.TableTensor.from_pandas(
    ...     pd.DataFrame({"color": ["red", "blue", "red"]}),
    ...     stypes={"color": "categorical"},
    ... )
    >>> table2 = sdm.TableTensor.from_pandas(
    ...     pd.DataFrame({"color": ["blue", "green", None]}),
    ...     stypes={"color": "categorical"},
    ... )
    >>> table1.categorical.categories[0].tolist()
    ['red', 'blue']
    >>> table2.categorical.categories[0].tolist()
    ['blue', 'green']
    >>> table2.categorical.code[:, 0].tolist()
    [0, 1, -1]
    >>> processor = AlignCategories().fit(table1)
    >>> table2 = processor.transform(table2)
    >>> table2.categorical.categories[0].tolist()
    ['red', 'blue']
    >>> table2.categorical.code[:, 0].tolist()
    [1, -1, -1]

    Here, ``"blue"`` changes from code ``0`` to ``1``, while unseen
    ``"green"`` and the missing value use ``-1``.
    """

    supported_stypes = frozenset({Stype.categorical})

    def __init__(
        self,
        sort_by: Literal["code", "frequency", "value"] = "code",
    ) -> None:
        super().__init__()
        self.sort_by = sort_by
        self._categories: tuple[tuple[Tensor, ...], ...] = ()

    def _fit_column(
        self,
        input_categories: Tensor,
        codes: Tensor,
        *,
        align_codes: bool,
    ) -> tuple[tuple[Tensor, ...], Tensor | None]:
        batch_size = codes.size(0)
        if input_categories.numel() == 0:
            aligned_codes = torch.full_like(codes, -1) if align_codes else None
            return (input_categories,) * batch_size, aligned_codes

        mask = codes >= 0
        indices = codes.clamp_min(0).long()
        # Count categories independently per batch: [B, N] -> [B, K].
        counts = codes.new_zeros((batch_size, input_categories.numel()))
        counts.scatter_add_(1, indices, mask.to(codes.dtype))

        if self.sort_by == "frequency":
            order = counts.argsort(dim=1, descending=True, stable=True)
            ordered_categories = input_categories
        elif self.sort_by == "value":
            if (
                input_categories.is_cuda
                and input_categories.dtype in _UNSIGNED_DTYPES
            ):
                key = input_categories.to(torch.int64)
                if input_categories.dtype == torch.uint64:
                    # Map unsigned integer order onto signed integer order.
                    key = key.bitwise_xor(torch.iinfo(torch.int64).min)
                perm = key.argsort()
                ordered_categories = input_categories.index_select(0, perm)
            else:
                ordered_categories, perm = input_categories.sort()
            order = perm.expand(batch_size, -1)
        else:
            assert self.sort_by == "code"
            order = torch.arange(
                input_categories.numel(),
                device=codes.device,
            ).expand(batch_size, -1)
            ordered_categories = input_categories

        observed = counts.gather(1, order) > 0
        # Only ragged vocabularies require per-batch materialization.
        fitted_categories = []
        for batch_index in range(batch_size):
            selected_indices = order[batch_index, observed[batch_index]]
            if (
                selected_indices.numel() == input_categories.numel()
                and self.sort_by != "frequency"
            ):
                fitted_categories.append(ordered_categories)
            elif (
                input_categories.dtype in _UNSIGNED_DTYPES
                and input_categories.is_cpu
            ):
                # PyTorch CPU index_select is not implemented for these dtypes.
                fitted_categories.append(input_categories[selected_indices])
            else:
                fitted_categories.append(
                    input_categories.index_select(0, selected_indices)
                )

        if not align_codes:
            return tuple(fitted_categories), None

        rank = observed.cumsum(dim=1, dtype=codes.dtype) - 1
        lookup = torch.full_like(counts, -1)
        lookup.scatter_(1, order, torch.where(observed, rank, -1))
        aligned_codes = torch.where(mask, lookup.gather(1, indices), -1)
        return tuple(fitted_categories), aligned_codes

    def _fit_columns(
        self,
        table: TableTensor,
        *,
        align_codes: bool,
    ) -> tuple[tuple[tuple[Tensor, ...], ...], Tensor | None]:
        codes = table.categorical.code
        if codes.dim() == 2:
            codes = codes.unsqueeze(0)
        aligned_codes = torch.full_like(codes, -1) if align_codes else None

        # Accumulate ragged vocabularies in batch-major order: [batch][column].
        categories_by_batch: list[list[Tensor]] = [
            [] for _ in range(codes.size(0))
        ]
        for column_index, input_categories in enumerate(
            table.categorical.categories
        ):
            fitted_categories, aligned_column_codes = self._fit_column(
                input_categories=input_categories,
                codes=codes[..., column_index],
                align_codes=align_codes,
            )
            if aligned_codes is not None:
                assert aligned_column_codes is not None
                aligned_codes[..., column_index] = aligned_column_codes
            for batch_categories, column_categories in zip(
                categories_by_batch,
                fitted_categories,
                strict=True,
            ):
                batch_categories.append(column_categories)

        fitted_categories = tuple(
            tuple(batch_categories) for batch_categories in categories_by_batch
        )
        return fitted_categories, aligned_codes

    def _fit_and_align(
        self,
        table: TableTensor,
    ) -> tuple[tuple[tuple[Tensor, ...], ...], tuple[TableTensor, ...]]:
        single_table = table.categorical.code.dim() == 2
        fitted_categories, aligned_codes = self._fit_columns(
            table,
            align_codes=True,
        )
        assert aligned_codes is not None
        aligned_tables = tuple(
            (table if single_table else table[batch_index]).replace_blocks(
                categorical=CategoricalTensor(
                    aligned_codes[batch_index],
                    categories=batch_categories,
                ),
            )
            for batch_index, batch_categories in enumerate(fitted_categories)
        )
        return fitted_categories, aligned_tables

    def _fit(
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        fitted_categories, _ = self._fit_columns(
            table,
            align_codes=False,
        )
        self._categories = fitted_categories

    def _fit_transform(
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> TableTensor:
        fitted_categories, aligned_tables = self._fit_and_align(table)
        self._categories = fitted_categories
        return aligned_tables[0]

    def _transform(self, table: TableTensor) -> TableTensor:
        return self._align_to_categories(table, self._categories)[0]

    @staticmethod
    def _member_table_ids(
        ensemble_table: EnsembleTable,
    ) -> tuple[int, ...]:
        group_offsets = []
        offset = 0
        for group in ensemble_table:
            group_offsets.append(offset)
            offset += group.size(0)

        # TODO: Replace this private access with a public EnsembleTable
        # operation returning flattened tables and member table IDs.
        return tuple(
            group_offsets[group_index] + position
            for group_index, position in ensemble_table._locations
        )

    def _fit_ensemble(
        self,
        ensemble_table: EnsembleTable,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        fitted_categories = []
        for group in ensemble_table:
            group_categories, _ = self._fit_columns(
                group,
                align_codes=False,
            )
            fitted_categories.extend(group_categories)
        self._categories = tuple(fitted_categories)

    def _fit_transform_ensemble(
        self,
        ensemble_table: EnsembleTable,
        *,
        generator: torch.Generator | None = None,
    ) -> EnsembleTable:
        fitted_categories = []
        aligned_tables = []
        for group in ensemble_table:
            group_categories, group_tables = self._fit_and_align(group)
            fitted_categories.extend(group_categories)
            aligned_tables.extend(group_tables)

        member_table_ids = self._member_table_ids(ensemble_table)
        output = EnsembleTable.from_tables(
            tables=aligned_tables,
            member_table_ids=member_table_ids,
        )
        self._categories = tuple(fitted_categories)
        return output

    def _transform_ensemble(
        self,
        ensemble_table: EnsembleTable,
    ) -> EnsembleTable:
        table_ids = self._member_table_ids(ensemble_table)
        aligned_tables = []
        offset = 0
        for group in ensemble_table:
            end = offset + group.size(0)
            aligned_tables.extend(
                self._align_to_categories(
                    group,
                    self._categories[offset:end],
                )
            )
            offset = end

        return EnsembleTable.from_tables(
            tables=aligned_tables,
            member_table_ids=table_ids,
        )

    @staticmethod
    def _category_lookup(
        input_categories: Tensor,
        fitted_categories: Tensor,
        codes: Tensor,
    ) -> Tensor:
        lookup = codes.new_full((input_categories.numel(),), -1)
        if fitted_categories.numel() == 0:
            return lookup

        if isinstance(input_categories, StringTensor):
            # TODO Join once with a column-index composite key.
            left_index, right_index = join_index(
                left_table=TableTensor(
                    columns={"id": ("id",)},
                    id=ColumnarTensor((input_categories,)),
                ),
                right_table=TableTensor(
                    columns={"id": ("id",)},
                    id=ColumnarTensor((fitted_categories,)),
                ),
                left_keys=["id"],
                right_keys=["id"],
                dtype=codes.dtype,
            )
            lookup[left_index] = right_index
        else:
            comparable_categories = input_categories
            sorted_categories = fitted_categories
            if (
                comparable_categories.dtype == torch.bool
                or comparable_categories.dtype in _UNSIGNED_DTYPES
            ):
                comparable_categories = comparable_categories.to(torch.int64)
                sorted_categories = sorted_categories.to(torch.int64)

            sorted_categories, perm = sorted_categories.sort()
            position = torch.searchsorted(
                sorted_categories,
                comparable_categories,
            )
            position = position.clamp(max=sorted_categories.numel() - 1)
            match = sorted_categories[position] == comparable_categories
            left_index = match.nonzero().view(-1)
            right_index = perm[position[left_index]]
            lookup[left_index] = right_index.to(codes.dtype)
        return lookup

    def _align_to_categories(
        self,
        table: TableTensor,
        fitted_categories: tuple[tuple[Tensor, ...], ...],
    ) -> tuple[TableTensor, ...]:
        codes = table.categorical.code
        single_table = codes.dim() == 2
        if single_table:
            codes = codes.unsqueeze(0)
        aligned_codes = torch.full_like(codes, -1)
        for column_index, input_categories in enumerate(
            table.categorical.categories
        ):
            if input_categories.numel() == 0:
                continue

            lookups = []
            lookup_by_categories: dict[int, Tensor] = {}
            for batch_categories in fitted_categories:
                column_categories = batch_categories[column_index]
                identity = id(column_categories)
                lookup = lookup_by_categories.get(identity)
                if lookup is None:
                    lookup = self._category_lookup(
                        input_categories,
                        column_categories,
                        codes,
                    )
                    lookup_by_categories[identity] = lookup
                lookups.append(lookup)

            column_codes = codes[..., column_index]
            mask = column_codes >= 0
            lookup = torch.stack(lookups)
            indices = column_codes.clamp_min(0).long()
            aligned_codes[..., column_index] = torch.where(
                mask,
                lookup.gather(1, indices),
                -1,
            )

        return tuple(
            (table if single_table else table[batch_index]).replace_blocks(
                categorical=CategoricalTensor(
                    aligned_codes[batch_index],
                    categories=batch_categories,
                ),
            )
            for batch_index, batch_categories in enumerate(fitted_categories)
        )

    def __repr__(self, *, indent: int = 0) -> str:
        return (
            f"{' ' * indent}{self.__class__.__name__}("
            f"sort_by={self.sort_by!r}"
            f")"
        )
