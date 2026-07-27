import torch
from torch import Tensor

from sdm import (
    CategoricalTensor,
    ColumnarTensor,
    StringTensor,
    Stype,
    TableTensor,
)
from sdm.processing import Processor
from sdm.relational.join import join_index

_UNSIGNED_DTYPES = frozenset({torch.uint16, torch.uint32, torch.uint64})


class AlignCategories(Processor):
    """Align categorical columns to vocabularies observed during fitting.

    Fitting keeps the observed category values for each column. Transforming
    remaps input codes by category value into those fitted vocabularies.
    Missing values and unseen categories are encoded as ``-1``.
    """

    supported_stypes = frozenset({Stype.categorical})

    def __init__(self) -> None:
        super().__init__()
        self._categories: tuple[Tensor, ...] = ()

    def _fit(
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        mask = table.categorical.isfinite()

        categories: list[Tensor] = []
        for i, category in enumerate(table.categorical.categories):
            index = table.categorical[..., i].view(-1)
            unique = index[mask[..., i]].unique()
            if unique.numel() == category.numel():
                pass
            elif category.dtype in _UNSIGNED_DTYPES and category.is_cpu:
                # PyTorch CPU index_select is not implemented for these dtypes.
                category = category[unique]
            else:
                category = category.index_select(0, unique)
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
            index = table.categorical[..., i].view(-1)
            unique, inverse = index[mask[..., i]].unique(return_inverse=True)
            if unique.numel() == category.numel():
                pass
            elif category.dtype in _UNSIGNED_DTYPES and category.is_cpu:
                # PyTorch CPU index_select is not implemented for these dtypes.
                category = category[unique]
            else:
                category = category.index_select(0, unique)
            categories.append(category)

            out[..., i][mask[..., i]] = inverse.to(out.dtype)

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
            index = table.categorical[..., i]
            unique, inverse = index[mask[..., i]].unique(return_inverse=True)
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
            out[..., i][mask[..., i]] = remapped[inverse]

        return table.replace_blocks(
            categorical=CategoricalTensor(out, categories=self._categories),
        )
