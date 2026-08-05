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
        self._categories_by_member: tuple[tuple[Tensor, ...], ...] = ()

    def _fit_category(
        self,
        category: Tensor,
        code: Tensor,
        *,
        return_inverse: bool,
    ) -> tuple[Tensor, Tensor | None]:
        perm: Tensor | None = None
        inverse: Tensor | None = None
        if self.sort_by == "frequency":
            if return_inverse:
                unique, inverse, count = code.unique(
                    return_inverse=True,
                    return_counts=True,
                )
            else:
                unique, count = code.unique(return_counts=True)
            perm = count.argsort(descending=True, stable=True)
            unique = unique[perm]
        else:
            if return_inverse:
                unique, inverse = code.unique(return_inverse=True)
            else:
                unique = code.unique()

        if unique.numel() != category.numel() or self.sort_by == "frequency":
            if category.dtype in _UNSIGNED_DTYPES and category.is_cpu:
                # PyTorch CPU index_select is not implemented for these dtypes.
                category = category[unique]
            else:
                category = category.index_select(0, unique)

        if self.sort_by == "value":
            if category.is_cuda and category.dtype in _UNSIGNED_DTYPES:
                key = category.to(torch.int64)
                if category.dtype == torch.uint64:
                    # Map unsigned integer order onto signed integer order:
                    key = key.bitwise_xor(torch.iinfo(torch.int64).min)
                perm = key.argsort()
                category = category.index_select(0, perm)
            else:
                category, perm = category.sort()

        if inverse is not None and perm is not None:
            inv_perm = torch.empty_like(perm)
            inv_perm[perm] = torch.arange(perm.numel(), device=perm.device)
            inverse = inv_perm[inverse]

        return category, inverse

    def _fit_categories(self, table: TableTensor) -> tuple[Tensor, ...]:
        mask = table.categorical.isfinite()

        categories: list[Tensor] = []
        for i, category in enumerate(table.categorical.categories):
            category, _ = self._fit_category(
                category=category,
                code=table.categorical[..., i].view(-1)[mask[..., i].view(-1)],
                return_inverse=False,
            )
            categories.append(category)

        return tuple(categories)

    def _fit_and_align(
        self,
        table: TableTensor,
    ) -> tuple[tuple[Tensor, ...], TableTensor]:
        mask = table.categorical.isfinite()
        out = torch.full_like(table.categorical, -1)

        categories: list[Tensor] = []
        for i, category in enumerate(table.categorical.categories):
            category, inverse = self._fit_category(
                category=category,
                code=table.categorical[..., i].view(-1)[mask[..., i].view(-1)],
                return_inverse=True,
            )
            categories.append(category)

            assert inverse is not None
            out[..., i].view(-1)[mask[..., i].view(-1)] = inverse.to(out.dtype)

        fitted_categories = tuple(categories)
        output = table.replace_blocks(
            categorical=CategoricalTensor(out, categories=fitted_categories),
        )
        return fitted_categories, output

    def _fit(
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        categories = self._fit_categories(table)
        self._categories_by_member = (categories,)

    def _fit_transform(
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> TableTensor:
        categories, output = self._fit_and_align(table)
        self._categories_by_member = (categories,)
        return output

    def _transform(self, table: TableTensor) -> TableTensor:
        return self._align_to_categories(
            table,
            self._categories_by_member[0],
        )

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
            self._fit_categories(group[position])
            for group in ensemble_table
            for position in range(group.size(0))
        )
        member_table_ids = self._member_table_ids(ensemble_table)
        self._categories_by_member = tuple(
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
            for position in range(group.size(0)):
                fitted_categories, table = self._fit_and_align(group[position])
                categories.append(fitted_categories)
                tables.append(table)

        member_table_ids = self._member_table_ids(ensemble_table)
        output = EnsembleTable.from_tables(
            tables=tables,
            member_table_ids=member_table_ids,
        )
        self._categories_by_member = tuple(
            categories[table_id] for table_id in member_table_ids
        )
        return output

    def _transform_ensemble(
        self,
        ensemble_table: EnsembleTable,
    ) -> EnsembleTable:
        if len(self._categories_by_member) != ensemble_table.num_members:
            raise RuntimeError(
                "AlignCategories must be fitted with the same number of "
                "ensemble members before transform."
            )

        route_id_by_categories: dict[int, int] = {}
        routes = []
        route_ids = []
        for categories in self._categories_by_member:
            identity = id(categories)
            route_id = route_id_by_categories.get(identity)
            if route_id is None:
                route_id = len(routes)
                route_id_by_categories[identity] = route_id
                routes.append(categories)
            route_ids.append(route_id)

        if len(routes) == 1:
            categories = routes[0]
            return ensemble_table.replace_groups(
                [
                    self._align_to_categories(group, categories)
                    for group in ensemble_table
                ]
            )

        if len(routes) == ensemble_table.num_members:
            return EnsembleTable.from_tables(
                tables=[
                    self._align_to_categories(
                        ensemble_table.table(member_id),
                        categories,
                    )
                    for member_id, categories in enumerate(
                        self._categories_by_member
                    )
                ],
                member_table_ids=range(ensemble_table.num_members),
            )

        member_ids_by_route = [[] for _ in routes]
        for member_id, route_id in enumerate(route_ids):
            member_ids_by_route[route_id].append(member_id)

        outputs = []
        for categories, member_ids in zip(
            routes,
            member_ids_by_route,
            strict=True,
        ):
            selected = ensemble_table.select_members(member_ids)
            outputs.append(
                selected.replace_groups(
                    [
                        self._align_to_categories(group, categories)
                        for group in selected
                    ]
                )
            )

        tables = []
        member_ids = []
        next_member_ids = [0] * len(routes)
        for route_id in route_ids:
            tables.append(outputs[route_id])
            member_ids.append(next_member_ids[route_id])
            next_member_ids[route_id] += 1

        return EnsembleTable.gather_members(
            tables=tables,
            member_ids=member_ids,
        )

    def _align_to_categories(
        self,
        table: TableTensor,
        categories: tuple[Tensor, ...],
    ) -> TableTensor:
        mask = table.categorical.isfinite()
        out = torch.full_like(table.categorical, -1)
        for i, (actual, expected) in enumerate(
            zip(table.categorical.categories, categories)
        ):
            if expected.numel() == 0:
                continue
            index = table.categorical[..., i].view(-1)
            unique, inverse = index[mask[..., i].view(-1)].unique(
                return_inverse=True,
            )
            if unique.numel() == 0:
                continue
            if unique.numel() == actual.numel():
                pass
            elif actual.dtype in _UNSIGNED_DTYPES and actual.is_cpu:
                # PyTorch CPU index_select is not implemented for these dtypes.
                actual = actual[unique]
            else:
                actual = actual.index_select(0, unique)

            if isinstance(actual, StringTensor):
                # TODO Run join once with column-index composite key.
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
                if (
                    actual.dtype == torch.bool
                    or actual.dtype in _UNSIGNED_DTYPES
                ):
                    actual = actual.to(torch.int64)
                    expected = expected.to(torch.int64)

                expected, perm = expected.sort()
                position = torch.searchsorted(expected, actual)
                position = position.clamp(max=expected.numel() - 1)
                match = expected[position] == actual
                left_index = match.nonzero().view(-1)
                right_index = perm[position[left_index]]

            remapped = out.new_full((actual.numel(),), fill_value=-1)
            remapped[left_index] = right_index.to(out.dtype)
            out[..., i].view(-1)[mask[..., i].view(-1)] = remapped[inverse]

        return table.replace_blocks(
            categorical=CategoricalTensor(out, categories=categories),
        )

    def __repr__(self, *, indent: int = 0) -> str:
        return (
            f"{' ' * indent}{self.__class__.__name__}("
            f"sort_by={self.sort_by!r}"
            f")"
        )
