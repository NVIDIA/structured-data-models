from typing import Literal, cast

import torch
from torch import Tensor

from sdm import (
    CategoricalTensor,
    StringTensor,
    Stype,
    TableTensor,
)
from sdm.processing import Processor
from sdm.tensor.categorical import _category
from sdm.tensor.string import _pairwise_equal

_UNSIGNED_DTYPES = frozenset({torch.uint16, torch.uint32, torch.uint64})


class AlignCategories(Processor):
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
        code = table.categorical.code
        mask = code >= 0

        categories: list[Tensor] = []
        for i, category in enumerate(table.categorical.categories):
            category, _ = self._fit_category(
                category=category,
                code=code[..., i].view(-1)[mask[..., i].view(-1)],
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
        code = table.categorical.code
        mask = code >= 0
        out = torch.full_like(code, -1)

        categories: list[Tensor] = []
        for i, category in enumerate(table.categorical.categories):
            category, inverse = self._fit_category(
                category=category,
                code=code[..., i].view(-1)[mask[..., i].view(-1)],
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
        code = table.categorical.code
        mask = code >= 0
        out = torch.full_like(code, -1)
        for i, expected in enumerate(self._categories):
            actual = _category(table.categorical, i)
            if expected.numel() == 0:
                continue
            index = code[..., i].view(-1)

            if isinstance(actual, StringTensor):
                match = _pairwise_equal(
                    actual,
                    cast(StringTensor, expected),
                )
                right_index = match.to(torch.int64).argmax(dim=1)
                remapped = torch.where(
                    match.any(dim=1),
                    right_index.to(out.dtype),
                    -1,
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
                remapped = torch.where(
                    match,
                    perm[position].to(out.dtype),
                    -1,
                )
            valid = mask[..., i].view(-1)
            remapped = torch.cat((remapped, remapped.new_full((1,), -1)))
            safe_index = torch.where(valid, index, actual.numel())
            out[..., i] = remapped[safe_index].view_as(out[..., i])

        return table.replace_blocks(
            categorical=CategoricalTensor(out, categories=self._categories),
        )

    def __repr__(self, *, indent: int = 0) -> str:
        return (
            f"{' ' * indent}{self.__class__.__name__}("
            f"sort_by={self.sort_by!r}"
            f")"
        )
