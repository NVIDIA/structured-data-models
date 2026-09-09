import math
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
from sdm.relational.join import join_index

_UNSIGNED_DTYPES = frozenset({torch.uint16, torch.uint32, torch.uint64})


class AlignCategories(EnsembleProcessor):
    """Align categorical columns to categories observed during fitting.

    A categorical column stores each value as an integer code indexing an
    ordered list of categories. Separately created tables can use different
    codes for the same value. Fitting learns the category list for each column,
    and transforming remaps another table to use it. Missing values and
    categories not retained during fitting receive code ``-1``.

    Leading batch dimensions are fitted independently and follow ordinary
    broadcasting during transform. Because a :class:`CategoricalTensor` has
    one vocabulary per column, a batched output uses the union of categories
    retained across batches and masks categories not retained by each batch.
    With ``sort_by="frequency"``, that shared vocabulary is ordered by total
    frequency across batches.

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

    handles_stypes = frozenset({Stype.categorical})
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
        self._retained: BufferList[BufferList[Tensor]] = BufferList()
        self._category_ids: tuple[int, ...] = ()

    def get_extra_state(self) -> tuple[int, ...]:
        r""":meta private:"""  # noqa: D415
        return self._category_ids

    def set_extra_state(self, state: object) -> None:
        r""":meta private:"""  # noqa: D415
        self._category_ids = cast(tuple[int, ...], state)

    def _fit_column(
        self,
        input_categories: Tensor,
        codes: Tensor,
        *,
        align_codes: bool,
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        batch_size = codes.size(0)
        if input_categories.numel() == 0:
            aligned_codes = torch.full_like(codes, -1) if align_codes else None
            retained = torch.empty(
                (batch_size, 0),
                dtype=torch.bool,
                device=codes.device,
            )
            return input_categories, retained, aligned_codes

        mask = codes >= 0
        indices = codes.clamp_min(0).long()
        # Count categories independently per batch: [B, N] -> [B, K].
        counts = codes.new_zeros((batch_size, input_categories.numel()))
        counts.scatter_add_(1, indices, mask.to(codes.dtype))

        if self.sort_by == "frequency":
            order = counts.sum(dim=0).argsort(
                descending=True,
                stable=True,
            )
        elif self.sort_by == "value":
            if (
                input_categories.is_cuda
                and input_categories.dtype in _UNSIGNED_DTYPES
            ):
                key = input_categories.to(torch.int64)
                if input_categories.dtype == torch.uint64:
                    # Map unsigned integer order onto signed integer order.
                    key = key.bitwise_xor(torch.iinfo(torch.int64).min)
                order = key.argsort()
            else:
                _, order = input_categories.sort()
        else:
            assert self.sort_by == "code"
            order = torch.arange(
                input_categories.numel(),
                device=codes.device,
            )

        retained = counts.index_select(1, order) >= self.min_frequency
        selected = retained.any(dim=0)
        selected_indices = order[selected]
        retained = retained[:, selected]
        if (
            selected_indices.numel() == input_categories.numel()
            and self.sort_by == "code"
        ):
            fitted_categories = input_categories
        elif (
            input_categories.dtype in _UNSIGNED_DTYPES
            and input_categories.is_cpu
        ):
            # A signed view preserves large unsigned values during gathering.
            signed_dtype = {
                torch.uint16: torch.int16,
                torch.uint32: torch.int32,
                torch.uint64: torch.int64,
            }[input_categories.dtype]
            fitted_categories = (
                input_categories.view(signed_dtype)
                .index_select(0, selected_indices)
                .view(input_categories.dtype)
            )
        else:
            fitted_categories = input_categories.index_select(
                0,
                selected_indices,
            )

        if not align_codes:
            return fitted_categories, retained, None

        aligned_codes = torch.full_like(codes, -1)
        if selected_indices.numel() > 0:
            lookup = codes.new_full((input_categories.numel(),), -1)
            lookup[selected_indices] = torch.arange(
                selected_indices.numel(),
                dtype=codes.dtype,
                device=codes.device,
            )
            candidate = lookup[indices]
            matched = candidate >= 0
            observed = retained.gather(1, candidate.clamp_min(0).long())
            aligned_codes = torch.where(
                mask & matched & observed,
                candidate,
                -1,
            )
        return fitted_categories, retained, aligned_codes

    def _fit_columns(
        self,
        table: TableTensor,
        *,
        align_codes: bool,
    ) -> tuple[tuple[Tensor, ...], tuple[Tensor, ...], Tensor | None]:
        codes = table.categorical.code
        batch_shape = tuple(codes.shape[:-2])
        batch_size = math.prod(batch_shape)
        flat_codes = codes.reshape(
            batch_size,
            codes.size(-2),
            codes.size(-1),
        )
        aligned_codes = (
            torch.full_like(flat_codes, -1) if align_codes else None
        )

        fitted_categories = []
        retained_by_column = []
        for column_index, input_categories in enumerate(
            table.categorical.categories
        ):
            column_categories, retained, aligned_column_codes = (
                self._fit_column(
                    input_categories=input_categories,
                    codes=flat_codes[..., column_index],
                    align_codes=align_codes,
                )
            )
            fitted_categories.append(column_categories)
            retained_by_column.append(
                retained.reshape(*batch_shape, retained.size(-1))
            )
            if aligned_codes is not None:
                assert aligned_column_codes is not None
                aligned_codes[..., column_index] = aligned_column_codes

        return (
            tuple(fitted_categories),
            tuple(retained_by_column),
            None
            if aligned_codes is None
            else aligned_codes.reshape(codes.shape),
        )

    def _fit_and_align(
        self,
        table: TableTensor,
    ) -> tuple[tuple[Tensor, ...], tuple[Tensor, ...], TableTensor]:
        fitted_categories, retained, aligned_codes = self._fit_columns(
            table,
            align_codes=True,
        )
        assert aligned_codes is not None
        aligned_table = table.replace_blocks(
            categorical=CategoricalTensor(
                aligned_codes,
                categories=fitted_categories,
            ),
        )
        return fitted_categories, retained, aligned_table

    def _store_states(
        self,
        categories: Sequence[Sequence[Tensor]],
        retained: Sequence[Sequence[Tensor]],
        category_ids: tuple[int, ...],
    ) -> None:
        self._categories = BufferList(
            BufferList(state_categories) for state_categories in categories
        )
        self._retained = BufferList(
            BufferList(state_retained) for state_retained in retained
        )
        self._category_ids = category_ids

    def _fit(
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        fitted_categories, retained, _ = self._fit_columns(
            table,
            align_codes=False,
        )
        self._store_states(
            categories=(fitted_categories,),
            retained=(retained,),
            category_ids=(0,),
        )

    def _fit_transform(
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> TableTensor:
        fitted_categories, retained, aligned_table = self._fit_and_align(table)
        self._store_states(
            categories=(fitted_categories,),
            retained=(retained,),
            category_ids=(0,),
        )
        return aligned_table

    def _transform(self, table: TableTensor) -> TableTensor:
        return self._align_to_categories(
            table,
            cast(Sequence[Tensor], self._categories[0]),
            cast(Sequence[Tensor], self._retained[0]),
        )

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
        retained_by_state = []
        for group in ensemble_table:
            for table_index in range(group.size(0)):
                categories, retained, _ = self._fit_columns(
                    group[table_index],
                    align_codes=False,
                )
                fitted_categories.append(categories)
                retained_by_state.append(retained)
        self._store_states(
            categories=fitted_categories,
            retained=retained_by_state,
            category_ids=self._member_table_ids(ensemble_table),
        )

    def _fit_transform_ensemble(
        self,
        ensemble_table: EnsembleTable,
        *,
        generator: torch.Generator | None = None,
    ) -> EnsembleTable:
        fitted_categories = []
        retained_by_state = []
        aligned_tables = []
        for group in ensemble_table:
            for table_index in range(group.size(0)):
                categories, retained, table = self._fit_and_align(
                    group[table_index]
                )
                fitted_categories.append(categories)
                retained_by_state.append(retained)
                aligned_tables.append(table)

        member_table_ids = self._member_table_ids(ensemble_table)
        output = EnsembleTable.from_tables(
            tables=aligned_tables,
            member_table_ids=member_table_ids,
        )
        self._store_states(
            categories=fitted_categories,
            retained=retained_by_state,
            category_ids=member_table_ids,
        )
        return output

    def _transform_ensemble(
        self,
        ensemble_table: EnsembleTable,
    ) -> EnsembleTable:
        if len(self._category_ids) != ensemble_table.num_members:
            raise RuntimeError(
                "AlignCategories must be fitted with the same number of "
                "ensemble members before transform."
            )

        aligned_tables = []
        member_table_ids = []
        aligned_table_ids: dict[tuple[int, int], int] = {}
        input_table_ids = self._member_table_ids(ensemble_table)
        for member_id, (input_table_id, category_id) in enumerate(
            zip(input_table_ids, self._category_ids, strict=True)
        ):
            key = (input_table_id, category_id)
            aligned_table_id = aligned_table_ids.get(key)
            if aligned_table_id is None:
                aligned_table_id = len(aligned_tables)
                aligned_table_ids[key] = aligned_table_id
                aligned_tables.append(
                    self._align_to_categories(
                        ensemble_table.table(member_id),
                        cast(
                            Sequence[Tensor],
                            self._categories[category_id],
                        ),
                        cast(
                            Sequence[Tensor],
                            self._retained[category_id],
                        ),
                    )
                )
            member_table_ids.append(aligned_table_id)

        return EnsembleTable.from_tables(
            tables=aligned_tables,
            member_table_ids=member_table_ids,
        )

    @staticmethod
    def _string_category_lookups(
        input_categories: tuple[StringTensor, ...],
        fitted_categories: tuple[Tensor, ...],
        codes: Tensor,
    ) -> tuple[Tensor, ...]:
        if not input_categories:
            return ()

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
                    (left_pair, torch.cat(input_categories)),
                ),
            ),
            right_table=TableTensor(
                columns={"id": ("pair", "value")},
                id=ColumnarTensor(
                    (right_pair, torch.cat(fitted_categories)),
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
        return lookup.split(
            [categories.numel() for categories in input_categories]
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
        fitted_categories: Sequence[Tensor],
        retained_by_column: Sequence[Tensor],
    ) -> TableTensor:
        codes = table.categorical.code
        batch_shape = tuple(codes.shape[:-2])
        fitted_batch_shape = (
            tuple(retained_by_column[0].shape[:-1])
            if retained_by_column
            else ()
        )
        try:
            output_batch_shape = torch.broadcast_shapes(
                fitted_batch_shape,
                batch_shape,
            )
        except RuntimeError:
            output_batch_shape = None
        if output_batch_shape != batch_shape:
            raise ValueError(
                "AlignCategories fitted batch shape must broadcast to the "
                f"transform batch shape; got {fitted_batch_shape} and "
                f"{batch_shape}."
            )
        if len(table.categorical.categories) != len(fitted_categories) or len(
            fitted_categories
        ) != len(retained_by_column):
            raise ValueError(
                "AlignCategories must be transformed with the fitted number "
                "of categorical columns."
            )

        batch_size = math.prod(batch_shape)
        flat_codes = codes.reshape(
            batch_size,
            codes.size(-2),
            codes.size(-1),
        )
        aligned_codes = torch.full_like(flat_codes, -1)
        lookups: list[Tensor | None] = [None] * len(fitted_categories)
        string_requests: list[tuple[int, StringTensor, Tensor]] = []
        for column_index, (input_categories, column_categories) in enumerate(
            zip(
                table.categorical.categories,
                fitted_categories,
                strict=True,
            )
        ):
            if input_categories.numel() == 0 or column_categories.numel() == 0:
                lookups[column_index] = codes.new_full(
                    (input_categories.numel(),),
                    -1,
                )
                continue
            if isinstance(input_categories, StringTensor) != isinstance(
                column_categories,
                StringTensor,
            ):
                raise NotImplementedError(
                    "Cannot align string categories with non-string categories"
                )
            if isinstance(input_categories, StringTensor):
                string_requests.append(
                    (column_index, input_categories, column_categories)
                )
                continue
            lookups[column_index] = self._category_lookup(
                input_categories,
                column_categories,
                codes,
            )

        string_lookups = self._string_category_lookups(
            input_categories=tuple(
                input_categories for _, input_categories, _ in string_requests
            ),
            fitted_categories=tuple(
                categories for _, _, categories in string_requests
            ),
            codes=codes,
        )
        for (column_index, _, _), lookup in zip(
            string_requests,
            string_lookups,
            strict=True,
        ):
            lookups[column_index] = lookup

        for column_index, (lookup, retained) in enumerate(
            zip(lookups, retained_by_column, strict=True)
        ):
            assert lookup is not None
            if lookup.numel() == 0 or retained.size(-1) == 0:
                continue
            column_codes = flat_codes[..., column_index]
            mask = column_codes >= 0
            indices = column_codes.clamp_min(0).long()
            candidate = lookup[indices]
            matched = candidate >= 0
            retained = retained.expand(*batch_shape, retained.size(-1))
            flat_retained = retained.reshape(batch_size, retained.size(-1))
            observed = flat_retained.gather(
                1,
                candidate.clamp_min(0).long(),
            )
            aligned_codes[..., column_index] = torch.where(
                mask & matched & observed,
                candidate,
                -1,
            )

        return table.replace_blocks(
            categorical=CategoricalTensor(
                aligned_codes.reshape(codes.shape),
                categories=tuple(fitted_categories),
            ),
        )

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
