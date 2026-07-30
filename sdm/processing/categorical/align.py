from typing import Literal, cast

import torch
from torch import Tensor
from typing_extensions import Self

from sdm import (
    CategoricalTensor,
    ColumnarTensor,
    StringTensor,
    Stype,
    TableTensor,
)
from sdm.processing.base import Processor
from sdm.processing.ensemble import VariableSchemaBatchMixin
from sdm.relational.join import join_index

_UNSIGNED_DTYPES = frozenset({torch.uint16, torch.uint32, torch.uint64})


class AlignCategories(Processor, VariableSchemaBatchMixin):
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
        self._batch_processors = torch.nn.ModuleList()

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

    def _transform(self, table: TableTensor) -> TableTensor:
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

    def _check_batch(self, table: TableTensor) -> None:
        self._check_supported_stypes(table)
        if table.dim() != 3:
            raise ValueError(
                "'AlignCategories' batch methods expect shape [V, R, C]."
            )

    def fit_batch(
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> Self:
        r"""Fit one categorical vocabulary per leading variant.

        Args:
            table: Variant batch with shape ``[V, R, C]``.
            generator: Optional pseudorandom number generator.
        """
        self._check_batch(table)
        self._batch_processors = torch.nn.ModuleList()
        for index in range(table.size(0)):
            processor = self.__class__(sort_by=self.sort_by)
            processor.fit(table[index], generator=generator)
            self._batch_processors.append(processor)
        return self

    def transform_batch(
        self,
        table: TableTensor,
    ) -> tuple[TableTensor, ...]:
        r"""Apply each fitted vocabulary to its leading variant.

        Args:
            table: Variant batch with shape ``[V, R, C]``.
        """
        self._check_batch(table)
        if table.size(0) != len(self._batch_processors):
            raise ValueError(
                "Expected the fitted number of leading variants "
                f"(got {table.size(0)})."
            )
        return tuple(
            cast(Processor, processor).transform(table[index])
            for index, processor in enumerate(self._batch_processors)
        )

    def __repr__(self, *, indent: int = 0) -> str:
        return (
            f"{' ' * indent}{self.__class__.__name__}("
            f"sort_by={self.sort_by!r}"
            f")"
        )
