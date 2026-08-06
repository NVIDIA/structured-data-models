from typing import Literal

import torch
from torch import Tensor

from sdm import Stype
from sdm.processing.ensemble import EnsembleProcessor
from sdm.relational.join import join_index
from sdm.tensor import (
    CategoricalTensor,
    ColumnarTensor,
    EnsembleTable,
    StringTensor,
    TableTensor,
)

_UNSIGNED_DTYPES = frozenset({torch.uint16, torch.uint32, torch.uint64})


class AlignCategories(EnsembleProcessor):
    """Align categorical columns to vocabularies observed during fitting.

    Fitting keeps the observed category values for each column. Transforming
    remaps input codes by category value into those fitted vocabularies.
    Missing values and unseen categories are encoded as ``-1``.

    Args:
        sort_by: How to order fitted category vocabularies.
            ``"code"`` keeps observed categories in original order.
            ``"frequency"`` orders observed categories by descending frequency.
            ``"value"`` orders observed categories by ascending value.
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
        return_aligned_codes: bool,
    ) -> tuple[tuple[Tensor, ...], Tensor | None]:
        batch_size = codes.size(0)
        if input_categories.numel() == 0:
            aligned_codes = (
                torch.full_like(codes, -1) if return_aligned_codes else None
            )
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

        if not return_aligned_codes:
            return tuple(fitted_categories), None

        rank = observed.cumsum(dim=1, dtype=codes.dtype) - 1
        lookup = torch.full_like(counts, -1)
        lookup.scatter_(1, order, torch.where(observed, rank, -1))
        aligned_codes = torch.where(mask, lookup.gather(1, indices), -1)
        return tuple(fitted_categories), aligned_codes

    def _fit_categories(
        self,
        table: TableTensor,
    ) -> tuple[tuple[Tensor, ...], ...]:
        codes = table.categorical.code
        batch_size = codes.size(0)
        if codes.dim() == 2:
            codes = codes.unsqueeze(0)

        # Accumulate ragged vocabularies in batch-major order: [batch][column].
        categories_by_batch: list[list[Tensor]] = [
            [] for _ in range(batch_size)
        ]
        for column_index, input_categories in enumerate(
            table.categorical.categories
        ):
            fitted_categories, _ = self._fit_column(
                input_categories=input_categories,
                codes=codes[..., column_index],
                return_aligned_codes=False,
            )
            for batch_categories, column_categories in zip(
                categories_by_batch,
                fitted_categories,
                strict=True,
            ):
                batch_categories.append(column_categories)

        return tuple(
            tuple(batch_categories) for batch_categories in categories_by_batch
        )

    def _fit_and_align(
        self,
        table: TableTensor,
    ) -> tuple[tuple[tuple[Tensor, ...], ...], tuple[TableTensor, ...]]:
        codes = table.categorical.code
        single_table = codes.dim() == 2
        if single_table:
            codes = codes.unsqueeze(0)
        aligned_codes = torch.full_like(codes, -1)

        # Each batch element gets an independent vocabulary and code mapping.
        categories_by_batch: list[list[Tensor]] = [
            [] for _ in range(codes.size(0))
        ]
        for column_index, input_categories in enumerate(
            table.categorical.categories
        ):
            fitted_categories, aligned_column_codes = self._fit_column(
                input_categories=input_categories,
                codes=codes[..., column_index],
                return_aligned_codes=True,
            )
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
        self._categories = self._fit_categories(table)

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
        groups = []
        table_id = 0
        for group in ensemble_table:
            num_tables = group.size(0)
            ids = torch.arange(
                table_id,
                table_id + num_tables,
                dtype=torch.float32,
            ).view(num_tables, 1, 1)
            groups.append(
                TableTensor.from_tensor(
                    tensor=ids,
                    columns=("table_id",),
                )
            )
            table_id += num_tables

        table_ids = ensemble_table.replace_groups(groups)
        # IDs are intentionally on CPU, so this does not synchronize CUDA.
        return tuple(
            int(table_ids.table(member_id).numerical[0, 0].item())
            for member_id in range(ensemble_table.num_members)
        )

    def _fit_ensemble(
        self,
        ensemble_table: EnsembleTable,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        fitted_categories = tuple(
            batch_categories
            for group in ensemble_table
            for batch_categories in self._fit_categories(group)
        )
        member_table_ids = self._member_table_ids(ensemble_table)
        self._categories = tuple(
            fitted_categories[table_id] for table_id in member_table_ids
        )

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
        self._categories = tuple(
            fitted_categories[table_id] for table_id in member_table_ids
        )
        return output

    def _transform_ensemble(
        self,
        ensemble_table: EnsembleTable,
    ) -> EnsembleTable:
        if len(self._categories) != ensemble_table.num_members:
            raise RuntimeError(
                "AlignCategories must be fitted with the same number of "
                "ensemble members before transform."
            )

        table_ids = self._member_table_ids(ensemble_table)
        fitted_categories_by_table: dict[int, tuple[Tensor, ...]] = {}
        for member_categories, table_id in zip(
            self._categories,
            table_ids,
            strict=True,
        ):
            previous = fitted_categories_by_table.get(table_id)
            if previous is not None and previous is not member_categories:
                aligned_tables = [
                    self._align_to_categories(
                        ensemble_table.table(member_id),
                        (member_categories,),
                    )[0]
                    for member_id, member_categories in enumerate(
                        self._categories
                    )
                ]
                return EnsembleTable.from_tables(
                    tables=aligned_tables,
                    member_table_ids=range(ensemble_table.num_members),
                )
            fitted_categories_by_table[table_id] = member_categories

        fitted_categories = tuple(
            fitted_categories_by_table[table_id]
            for table_id in range(len(fitted_categories_by_table))
        )
        aligned_tables = []
        offset = 0
        for group in ensemble_table:
            end = offset + group.size(0)
            aligned_tables.extend(
                self._align_to_categories(
                    group,
                    fitted_categories[offset:end],
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
