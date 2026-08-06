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

    def _fit_category(
        self,
        category: Tensor,
        code: Tensor,
        *,
        return_aligned: bool,
    ) -> tuple[tuple[Tensor, ...], Tensor | None]:
        num_tables = code.size(0)
        if category.numel() == 0:
            aligned = torch.full_like(code, -1) if return_aligned else None
            return (category,) * num_tables, aligned

        mask = code >= 0
        index = code.clamp_min(0).long()
        count = code.new_zeros((num_tables, category.numel()))
        count.scatter_add_(1, index, mask.to(code.dtype))

        if self.sort_by == "frequency":
            order = count.argsort(dim=1, descending=True, stable=True)
            ordered_category = category
        elif self.sort_by == "value":
            if category.is_cuda and category.dtype in _UNSIGNED_DTYPES:
                key = category.to(torch.int64)
                if category.dtype == torch.uint64:
                    # Map unsigned integer order onto signed integer order.
                    key = key.bitwise_xor(torch.iinfo(torch.int64).min)
                perm = key.argsort()
                ordered_category = category.index_select(0, perm)
            else:
                ordered_category, perm = category.sort()
            order = perm.expand(num_tables, -1)
        else:
            assert self.sort_by == "code"
            order = torch.arange(
                category.numel(),
                device=code.device,
            ).expand(num_tables, -1)
            ordered_category = category

        present = count.gather(1, order) > 0
        categories = []
        for table_id in range(num_tables):
            selected = order[table_id, present[table_id]]
            if (
                selected.numel() == category.numel()
                and self.sort_by != "frequency"
            ):
                categories.append(ordered_category)
            elif category.dtype in _UNSIGNED_DTYPES and category.is_cpu:
                # PyTorch CPU index_select is not implemented for these dtypes.
                categories.append(category[selected])
            else:
                categories.append(category.index_select(0, selected))

        if not return_aligned:
            return tuple(categories), None

        rank = present.cumsum(dim=1, dtype=code.dtype) - 1
        lookup = torch.full_like(count, -1)
        lookup.scatter_(1, order, torch.where(present, rank, -1))
        aligned = torch.where(mask, lookup.gather(1, index), -1)
        return tuple(categories), aligned

    def _fit_categories(
        self,
        table: TableTensor,
    ) -> tuple[tuple[Tensor, ...], ...]:
        code = table.categorical.code
        if code.dim() == 2:
            code = code.unsqueeze(0)

        categories: list[list[Tensor]] = [[] for _ in range(code.size(0))]
        for i, category in enumerate(table.categorical.categories):
            fitted, _ = self._fit_category(
                category=category,
                code=code[..., i],
                return_aligned=False,
            )
            for table_categories, fitted_category in zip(
                categories,
                fitted,
                strict=True,
            ):
                table_categories.append(fitted_category)

        return tuple(tuple(category) for category in categories)

    def _fit_and_align(
        self,
        table: TableTensor,
    ) -> tuple[tuple[tuple[Tensor, ...], ...], tuple[TableTensor, ...]]:
        code = table.categorical.code
        single_table = code.dim() == 2
        if single_table:
            code = code.unsqueeze(0)
        out = torch.full_like(code, -1)

        categories: list[list[Tensor]] = [[] for _ in range(code.size(0))]
        for i, category in enumerate(table.categorical.categories):
            fitted, aligned = self._fit_category(
                category=category,
                code=code[..., i],
                return_aligned=True,
            )
            assert aligned is not None
            out[..., i] = aligned
            for table_categories, fitted_category in zip(
                categories,
                fitted,
                strict=True,
            ):
                table_categories.append(fitted_category)

        fitted_categories = tuple(tuple(category) for category in categories)
        outputs = tuple(
            (table if single_table else table[table_id]).replace_blocks(
                categorical=CategoricalTensor(
                    out[table_id],
                    categories=table_categories,
                ),
            )
            for table_id, table_categories in enumerate(fitted_categories)
        )
        return fitted_categories, outputs

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
        categories, outputs = self._fit_and_align(table)
        self._categories = categories
        return outputs[0]

    def _transform(self, table: TableTensor) -> TableTensor:
        return self._align_tables(table, self._categories)[0]

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
        categories = tuple(
            fitted
            for group in ensemble_table
            for fitted in self._fit_categories(group)
        )
        member_table_ids = self._member_table_ids(ensemble_table)
        self._categories = tuple(
            categories[table_id] for table_id in member_table_ids
        )

    def _fit_transform_ensemble(
        self,
        ensemble_table: EnsembleTable,
        *,
        generator: torch.Generator | None = None,
    ) -> EnsembleTable:
        categories = []
        tables = []
        for group in ensemble_table:
            fitted, outputs = self._fit_and_align(group)
            categories.extend(fitted)
            tables.extend(outputs)

        member_table_ids = self._member_table_ids(ensemble_table)
        output = EnsembleTable.from_tables(
            tables=tables,
            member_table_ids=member_table_ids,
        )
        self._categories = tuple(
            categories[table_id] for table_id in member_table_ids
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

        states = []
        state_ids = []
        state_id_by_categories: dict[int, int] = {}
        for categories in self._categories:
            identity = id(categories)
            state_id = state_id_by_categories.get(identity)
            if state_id is None:
                state_id = len(states)
                state_id_by_categories[identity] = state_id
                states.append(categories)
            state_ids.append(state_id)

        physical_tables = tuple(
            group[position]
            for group in ensemble_table
            for position in range(group.size(0))
        )
        table_ids = self._member_table_ids(ensemble_table)
        state_id_by_table: dict[int, int] = {}
        for state_id, table_id in zip(state_ids, table_ids, strict=True):
            previous = state_id_by_table.setdefault(table_id, state_id)
            if previous != state_id:
                break
        else:
            categories = tuple(
                states[state_id_by_table[table_id]]
                for table_id in range(len(physical_tables))
            )
            outputs = []
            offset = 0
            for group in ensemble_table:
                end = offset + group.size(0)
                outputs.extend(
                    self._align_tables(group, categories[offset:end])
                )
                offset = end
            return EnsembleTable.from_tables(
                tables=outputs,
                member_table_ids=table_ids,
            )

        route_id_by_state_and_table: dict[tuple[int, int], int] = {}
        routes = []
        route_ids = []
        for state_id, table_id in zip(state_ids, table_ids, strict=True):
            key = (state_id, table_id)
            route_id = route_id_by_state_and_table.get(key)
            if route_id is None:
                route_id = len(routes)
                route_id_by_state_and_table[key] = route_id
                routes.append(key)
            route_ids.append(route_id)

        # One physical query table needs multiple outputs when its members
        # learned different vocabularies. Pack those routes so their row data
        # can still be aligned together.
        routed = EnsembleTable.from_tables(
            tables=[physical_tables[table_id] for _, table_id in routes],
            member_table_ids=range(len(routes)),
        )
        routed_table_ids = self._member_table_ids(routed)
        categories: list[tuple[Tensor, ...]] = [()] * len(routes)
        for route_id, table_id in enumerate(routed_table_ids):
            state_id, _ = routes[route_id]
            categories[table_id] = states[state_id]

        outputs = []
        offset = 0
        for group in routed:
            end = offset + group.size(0)
            outputs.extend(
                self._align_tables(group, tuple(categories[offset:end]))
            )
            offset = end

        return EnsembleTable.from_tables(
            tables=outputs,
            member_table_ids=[
                routed_table_ids[route_id] for route_id in route_ids
            ],
        )

    def _align_tables(
        self,
        table: TableTensor,
        categories: tuple[tuple[Tensor, ...], ...],
    ) -> tuple[TableTensor, ...]:
        code = table.categorical.code
        single_table = code.dim() == 2
        if single_table:
            code = code.unsqueeze(0)
        out = torch.full_like(code, -1)
        for i, actual in enumerate(table.categorical.categories):
            if actual.numel() == 0:
                continue
            values = code[..., i]
            mask = values >= 0
            lookups = []
            lookup_by_categories: dict[int, Tensor] = {}
            for fitted_categories in categories:
                expected = fitted_categories[i]
                identity = id(expected)
                lookup = lookup_by_categories.get(identity)
                if lookup is None:
                    lookup = out.new_full((actual.numel(),), -1)
                    if expected.numel() > 0:
                        if isinstance(actual, StringTensor):
                            # TODO Join once with a column-index composite key.
                            left_index, right_index = join_index(
                                left_table=TableTensor(
                                    columns={"id": ("id",)},
                                    id=ColumnarTensor((actual,)),
                                ),
                                right_table=TableTensor(
                                    columns={"id": ("id",)},
                                    id=ColumnarTensor((expected,)),
                                ),
                                left_keys=["id"],
                                right_keys=["id"],
                                dtype=out.dtype,
                            )
                        else:
                            comparable = actual
                            fitted = expected
                            if (
                                comparable.dtype == torch.bool
                                or comparable.dtype in _UNSIGNED_DTYPES
                            ):
                                comparable = comparable.to(torch.int64)
                                fitted = fitted.to(torch.int64)

                            fitted, perm = fitted.sort()
                            position = torch.searchsorted(fitted, comparable)
                            position = position.clamp(max=fitted.numel() - 1)
                            match = fitted[position] == comparable
                            left_index = match.nonzero().view(-1)
                            right_index = perm[position[left_index]]

                        lookup[left_index] = right_index.to(out.dtype)
                    lookup_by_categories[identity] = lookup
                lookups.append(lookup)

            lookup = torch.stack(lookups)
            index = values.clamp_min(0).long()
            out[..., i] = torch.where(
                mask,
                lookup.gather(1, index),
                -1,
            )

        return tuple(
            (table if single_table else table[table_id]).replace_blocks(
                categorical=CategoricalTensor(
                    out[table_id],
                    categories=table_categories,
                ),
            )
            for table_id, table_categories in enumerate(categories)
        )

    def __repr__(self, *, indent: int = 0) -> str:
        return (
            f"{' ' * indent}{self.__class__.__name__}("
            f"sort_by={self.sort_by!r}"
            f")"
        )
