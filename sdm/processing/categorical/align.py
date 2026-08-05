from typing import Literal, cast

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
        self._categories: tuple[Tensor, ...] = ()
        self.processors = torch.nn.ModuleList()
        self._member_processor_ids: tuple[int, ...] = ()

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

    def _fit(
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        self.processors = torch.nn.ModuleList()
        self._member_processor_ids = ()
        mask = table.categorical.isfinite()

        categories: list[Tensor] = []
        for i, category in enumerate(table.categorical.categories):
            category, _ = self._fit_category(
                category=category,
                code=table.categorical[..., i].view(-1)[mask[..., i].view(-1)],
                return_inverse=False,
            )
            categories.append(category)

        self._categories = tuple(categories)

    def _fit_transform(
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> TableTensor:
        self.processors = torch.nn.ModuleList()
        self._member_processor_ids = ()
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

        self._categories = tuple(categories)

        return table.replace_blocks(
            categorical=CategoricalTensor(out, categories=self._categories),
        )

    def _fit_ensemble(
        self,
        ensemble_table: EnsembleTable,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        processors = torch.nn.ModuleList()
        member_processor_ids = []
        fitted: dict[tuple[int, int], int] = {}

        for member_id in range(ensemble_table.num_members):
            location = ensemble_table._member_location(member_id)
            processor_id = fitted.get(location)
            if processor_id is None:
                processor = self.__class__(sort_by=self.sort_by)
                processor.fit(
                    ensemble_table.table(member_id),
                    generator=generator,
                )
                processor_id = len(processors)
                fitted[location] = processor_id
                processors.append(processor)
            member_processor_ids.append(processor_id)

        self.processors = processors
        self._member_processor_ids = tuple(member_processor_ids)

    def _fit_transform_ensemble(
        self,
        ensemble_table: EnsembleTable,
        *,
        generator: torch.Generator | None = None,
    ) -> EnsembleTable:
        processors = torch.nn.ModuleList()
        member_processor_ids = []
        tables = []
        member_table_ids = []
        fitted: dict[tuple[int, int], int] = {}

        for member_id in range(ensemble_table.num_members):
            location = ensemble_table._member_location(member_id)
            processor_id = fitted.get(location)
            if processor_id is None:
                processor = self.__class__(sort_by=self.sort_by)
                table = processor.fit_transform(
                    ensemble_table.table(member_id),
                    generator=generator,
                )
                processor_id = len(processors)
                fitted[location] = processor_id
                processors.append(processor)
                tables.append(table)
            member_processor_ids.append(processor_id)
            member_table_ids.append(processor_id)

        output = EnsembleTable.from_tables(
            tables=tables,
            member_table_ids=member_table_ids,
        )
        self.processors = processors
        self._member_processor_ids = tuple(member_processor_ids)
        return output

    def _transform_ensemble(
        self,
        ensemble_table: EnsembleTable,
    ) -> EnsembleTable:
        if len(self._member_processor_ids) != ensemble_table.num_members:
            raise RuntimeError(
                "AlignCategories must be fitted with the same number of "
                "ensemble members before transform."
            )

        tables = []
        member_table_ids = []
        transformed: dict[tuple[tuple[int, int], int], int] = {}
        for member_id, processor_id in enumerate(self._member_processor_ids):
            key = (
                ensemble_table._member_location(member_id),
                processor_id,
            )
            table_id = transformed.get(key)
            if table_id is None:
                processor = cast(
                    AlignCategories,
                    self.processors[processor_id],
                )
                table_id = len(tables)
                transformed[key] = table_id
                tables.append(
                    processor.transform(ensemble_table.table(member_id))
                )
            member_table_ids.append(table_id)

        return EnsembleTable.from_tables(
            tables=tables,
            member_table_ids=member_table_ids,
        )

    def _transform(self, table: TableTensor) -> TableTensor:
        if len(self.processors) > 0:
            raise RuntimeError(
                "'AlignCategories' was fitted for an ensemble; use "
                "'transform_ensemble' instead of 'transform'."
            )
        mask = table.categorical.isfinite()
        out = torch.full_like(table.categorical, -1)
        for i, (actual, expected) in enumerate(
            zip(table.categorical.categories, self._categories)
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
            categorical=CategoricalTensor(out, categories=self._categories),
        )

    def __repr__(self, *, indent: int = 0) -> str:
        return (
            f"{' ' * indent}{self.__class__.__name__}("
            f"sort_by={self.sort_by!r}"
            f")"
        )
