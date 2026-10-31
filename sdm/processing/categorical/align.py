# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
from typing import Literal, cast

import torch
from torch import Tensor

from sdm import (
    CategoricalTensor,
    ColumnarTensor,
    EnsembleTable,
    StringTensor,
    Stype,
    TableTensor,
)
from sdm.nn._buffer import BufferList
from sdm.processing import EnsembleProcessor
from sdm.processing.categorical._string_lookup import (
    _max_length,
    _native_string_lookups,
)
from sdm.relational.join import join_index
from sdm.tensor.categorical import _iter_categories
from sdm.tensor.var_len import _cat, _contiguous

_UNSIGNED_DTYPES = frozenset({torch.uint16, torch.uint32, torch.uint64})


class AlignCategories(EnsembleProcessor):
    """Align categorical columns to categories observed during fitting.

    A categorical column stores each value as an integer code indexing an
    ordered list of categories. Separately created tables can use different
    codes for the same value. Fitting learns the category list for each column,
    and transforming remaps another table to use it. Missing values and
    categories not retained during fitting receive code ``-1``.

    Args:
        sort_by: How to order fitted categories.
            ``"code"`` keeps observed categories in original order.
            ``"frequency"`` orders observed categories by descending frequency.
            ``"value"`` orders observed categories by ascending value.
        min_frequency: Minimum number of observations required to retain a
            category. Values of rarer categories receive code ``-1``.
            Must be positive.

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

    handles_stypes = (Stype.categorical,)
    requires_fit = True

    def __init__(
        self,
        sort_by: Literal["code", "frequency", "value"] = "code",
        *,
        min_frequency: int = 1,
    ) -> None:
        super().__init__()
        if min_frequency <= 0:
            raise ValueError("min_frequency must be positive")
        self.sort_by = sort_by
        self.min_frequency = min_frequency
        self._categories: BufferList[BufferList[Tensor]] = BufferList()
        # Byte length of the longest fitted string label, which sizes the
        # keys of native string lookups. It derives from the fitted
        # categories, and loading them measures it again.
        self._max_string_length = 0
        self.register_load_state_dict_post_hook(_measure_strings)

    def _set_categories(
        self, fitted_categories: Sequence[Sequence[Tensor]]
    ) -> None:
        self._categories = BufferList(
            BufferList(categories) for categories in fitted_categories
        )
        _measure_strings(self)

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

        observed = counts.gather(1, order) >= self.min_frequency
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
            _iter_categories(table.categorical)
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
        self._set_categories(fitted_categories)

    def _fit_transform(
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> TableTensor:
        fitted_categories, aligned_tables = self._fit_and_align(table)
        self._set_categories(fitted_categories)
        return aligned_tables[0]

    def _transform(self, table: TableTensor) -> TableTensor:
        return self._align_to_categories(
            table,
            cast(Sequence[Sequence[Tensor]], self._categories),
        )[0]

    @staticmethod
    def _member_table_ids(
        ensemble_table: EnsembleTable,
    ) -> tuple[int, ...]:
        group_offsets = []
        offset = 0
        for group in ensemble_table._iter_groups():
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
        for group in ensemble_table._iter_groups():
            group_categories, _ = self._fit_columns(
                group,
                align_codes=False,
            )
            fitted_categories.extend(group_categories)
        self._set_categories(fitted_categories)

    def _fit_transform_ensemble(
        self,
        ensemble_table: EnsembleTable,
        *,
        generator: torch.Generator | None = None,
    ) -> EnsembleTable:
        fitted_categories = []
        aligned_tables = []
        for group in ensemble_table._iter_groups():
            group_categories, group_tables = self._fit_and_align(group)
            fitted_categories.extend(group_categories)
            aligned_tables.extend(group_tables)

        member_table_ids = self._member_table_ids(ensemble_table)
        output = ensemble_table.replace_tables(
            tables=aligned_tables,
            member_table_ids=member_table_ids,
        )
        self._set_categories(fitted_categories)
        return output

    def _transform_ensemble(
        self,
        ensemble_table: EnsembleTable,
    ) -> EnsembleTable:
        table_ids = self._member_table_ids(ensemble_table)
        aligned_tables = []
        offset = 0
        for group in ensemble_table._iter_groups():
            end = offset + group.size(0)
            aligned_tables.extend(
                self._align_to_categories(
                    group,
                    tuple(
                        cast(Sequence[Tensor], self._categories[index])
                        for index in range(offset, end)
                    ),
                )
            )
            offset = end

        return ensemble_table.replace_tables(
            tables=aligned_tables,
            member_table_ids=table_ids,
        )

    def _string_category_lookups(
        self,
        input_categories: tuple[StringTensor, ...],
        fitted_categories: tuple[Tensor, ...],
        codes: Tensor,
    ) -> tuple[Tensor, ...]:
        if not input_categories:
            return ()
        right = cast(tuple[StringTensor, ...], fitted_categories)
        if codes.is_cuda:
            return _native_string_lookups(
                left=input_categories,
                right=right,
                max_length=self._max_string_length,
                dtype=codes.dtype,
            )
        lookup = _string_category_lookup(
            input_categories=input_categories,
            fitted_categories=right,
            codes=codes,
        )
        return lookup.split(
            [category.numel() for category in input_categories]
        )

    @staticmethod
    def _category_lookup(
        input_categories: Tensor,
        fitted_categories: Tensor,
        codes: Tensor,
    ) -> Tensor:
        if fitted_categories.numel() == 0:
            return codes.new_full((input_categories.numel(),), -1)

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
        return torch.where(match, perm[position].to(codes.dtype), -1)

    def _align_to_categories(
        self,
        table: TableTensor,
        fitted_categories: Sequence[Sequence[Tensor]],
    ) -> tuple[TableTensor, ...]:
        codes = table.categorical.code
        single_table = codes.dim() == 2
        if single_table:
            codes = codes.unsqueeze(0)
        lookups_by_column: list[dict[int, Tensor]] = [
            {} for _ in range(codes.size(-1))
        ]
        string_requests: list[tuple[int, int, StringTensor, Tensor]] = []
        for column_index, input_categories in enumerate(
            _iter_categories(table.categorical)
        ):
            lookup_by_categories = lookups_by_column[column_index]
            pending_categories: set[int] = set()
            for batch_categories in fitted_categories:
                column_categories = batch_categories[column_index]
                identity = id(column_categories)
                lookup = lookup_by_categories.get(identity)
                if lookup is not None or identity in pending_categories:
                    continue
                if column_categories.numel() == 0:
                    lookup_by_categories[identity] = codes.new_full(
                        (input_categories.numel(),),
                        -1,
                    )
                    continue
                if isinstance(input_categories, StringTensor) != isinstance(
                    column_categories, StringTensor
                ):
                    # Only an all-missing query column, whose categories
                    # are empty and untyped, may differ in type. Compiled
                    # code skips the check: size-oblivious tracing cannot
                    # tell an empty dimension from a nonempty one.
                    if (
                        not torch.compiler.is_compiling()
                        and input_categories.numel() != 0
                    ):
                        raise NotImplementedError(
                            "Cannot align string categories with non-string "
                            "categories"
                        )
                    lookup_by_categories[identity] = codes.new_full(
                        (input_categories.numel(),), -1
                    )
                    continue
                if isinstance(input_categories, StringTensor):
                    pending_categories.add(identity)
                    string_requests.append(
                        (
                            column_index,
                            identity,
                            input_categories,
                            column_categories,
                        )
                    )
                    continue
                lookup_by_categories[identity] = self._category_lookup(
                    input_categories,
                    column_categories,
                    codes,
                )

        if codes.size(-1) == 0:
            aligned_codes = torch.full_like(codes, -1)
        else:
            aligned_codes = self._align_columns(
                codes, fitted_categories, lookups_by_column, string_requests
            )

        return tuple(
            (table if single_table else table[batch_index]).replace_blocks(
                categorical=CategoricalTensor(
                    aligned_codes[batch_index],
                    categories=tuple(batch_categories),
                ),
            )
            for batch_index, batch_categories in enumerate(fitted_categories)
        )

    def _align_columns(
        self,
        codes: Tensor,
        fitted_categories: Sequence[Sequence[Tensor]],
        lookups_by_column: list[dict[int, Tensor]],
        string_requests: list[tuple[int, int, StringTensor, Tensor]],
    ) -> Tensor:
        """Remap ``[B, N, C]`` codes of columns whose lookups are pending.

        Lookups are released once they are joined into one table.
        """
        string_lookups = self._string_category_lookups(
            input_categories=tuple(
                input_categories
                for _, _, input_categories, _ in string_requests
            ),
            fitted_categories=tuple(
                fitted_categories
                for _, _, _, fitted_categories in string_requests
            ),
            codes=codes,
        )
        for (column_index, identity, _, _), lookup in zip(
            string_requests,
            string_lookups,
            strict=True,
        ):
            lookups_by_column[column_index][identity] = lookup
        del string_lookups

        # One gather over all columns of a [B, T] lookup. Each column's
        # block ends with -1 for missing codes, so empty vocabularies need no
        # size-dependent branch, and no strided column views are traced.
        lookups = [
            torch.stack(
                [
                    lookups_by_column[column_index][
                        id(categories[column_index])
                    ]
                    for categories in fitted_categories
                ]
            )
            for column_index in range(codes.size(-1))
        ]
        for lookup_by_categories in lookups_by_column:
            lookup_by_categories.clear()
        missing = codes.new_full((len(fitted_categories), 1), -1)
        lookup_table = torch.cat(
            [x for lookup in lookups for x in (lookup, missing)], 1
        )
        # Column c owns lookup positions [start_c, end_c]; end_c is -1.
        # Compiled code keeps a wide index to avoid a guard on table size.
        index_dtype = torch.int64
        if (
            not torch.compiler.is_compiling()
            and lookup_table.numel() < torch.iinfo(torch.int32).max
        ):
            index_dtype = torch.int32
        widths = torch.tensor(
            [lookup.size(1) + 1 for lookup in lookups],
            dtype=index_dtype,
            device=codes.device,
        )
        del lookups, missing
        # First position of each column's block in the flat table.
        starts = widths.cumsum(0, dtype=index_dtype) - widths
        if codes.size(0) > 1:
            batches = torch.arange(
                codes.size(0), dtype=index_dtype, device=codes.device
            )
            starts = starts + (batches * lookup_table.size(1))[:, None, None]
        table = lookup_table.flatten()

        def gather(chunk: Tensor) -> Tensor:
            # The narrow index is updated in place to keep temporaries small.
            index = torch.where(chunk >= 0, chunk, widths - 1).to(index_dtype)
            index += starts
            return table.index_select(0, index.flatten()).view(chunk.shape)

        # Eager code gathers chunks of rows, so the index stays small next to
        # the output; compiled code fuses the index into a single gather.
        if torch.compiler.is_compiling():
            return gather(codes)
        rows = max(1, _GATHER_CODES // (codes.size(0) * codes.size(-1)))
        if codes.size(1) <= rows:
            return gather(codes)
        output = torch.empty_like(codes)
        for start in range(0, codes.size(1), rows):
            output[:, start : start + rows] = gather(
                codes[:, start : start + rows]
            )
        return output

    def __repr__(self, *, indent: int = 0) -> str:
        arguments = []
        if self.sort_by != "code":
            arguments.append(f"sort_by={self.sort_by!r}")
        if self.min_frequency != 1:
            arguments.append(f"min_frequency={self.min_frequency!r}")
        if not arguments:
            return super().__repr__(indent=indent)
        return (
            f"{' ' * indent}{self.__class__.__name__}({', '.join(arguments)})"
        )


def _measure_strings(processor: AlignCategories, *_: object) -> None:
    """Record the byte length of the longest fitted string label."""
    processor._max_string_length = _max_length(
        category
        for categories in processor._categories
        for category in categories
        if isinstance(category, StringTensor)
    )


# Eager code gathers aligned codes in chunks of about this many codes.
_GATHER_CODES = 1 << 14


def _concatenate_strings(categories: tuple[StringTensor, ...]) -> StringTensor:
    contiguous = tuple(
        category if category.is_contiguous() else _contiguous(category)
        for category in categories
    )
    return cast(StringTensor, _cat(contiguous))


def _string_category_lookup(
    input_categories: tuple[StringTensor, ...],
    fitted_categories: tuple[StringTensor, ...],
    codes: Tensor,
) -> Tensor:
    # Pair IDs for each input category: (total_input_categories,).
    left_pair = torch.cat(
        [
            codes.new_full((categories.numel(),), pair_index)
            for pair_index, categories in enumerate(input_categories)
        ]
    )
    # Pair IDs for each fitted category: (total_fitted_categories,).
    right_pair = torch.cat(
        [
            codes.new_full((categories.numel(),), pair_index)
            for pair_index, categories in enumerate(fitted_categories)
        ]
    )
    left_index, right_index = join_index(
        left_table=TableTensor(
            columns={"id": ("pair", "value")},
            id=ColumnarTensor(
                (left_pair, _concatenate_strings(input_categories)),
            ),
        ),
        right_table=TableTensor(
            columns={"id": ("pair", "value")},
            id=ColumnarTensor(
                (right_pair, _concatenate_strings(fitted_categories)),
            ),
        ),
        left_keys=["pair", "value"],
        right_keys=["pair", "value"],
        dtype=codes.dtype,
    )
    fitted_codes = torch.cat(
        [
            torch.arange(
                categories.numel(),
                dtype=codes.dtype,
                device=codes.device,
            )
            for categories in fitted_categories
        ]
    )
    # Fitted code for each input category: (total_input_categories,).
    lookup = codes.new_full(
        (sum(categories.numel() for categories in input_categories),),
        -1,
    )
    lookup[left_index] = fitted_codes[right_index]
    return lookup
