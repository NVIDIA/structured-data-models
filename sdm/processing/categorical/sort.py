import torch
from torch import Tensor

from sdm import CategoricalTensor, StringTensor, Stype
from sdm.processing.base import Processor
from sdm.processing.categorical._categorical import _check_categorical_codes
from sdm.tensor import TableTensor

_HOST_SORTED_DTYPES = frozenset({torch.uint16, torch.uint32, torch.uint64})


class SortCategories(Processor):
    """Sort category vocabularies by value and remap their integer codes.

    Codes and their corresponding category vectors are reordered together, so
    decoded values remain unchanged. Negative codes represent missing values
    and are preserved unchanged. Only categorical columns are supported; use
    :class:`~sdm.processing.StypeDispatch` to apply this processor to the
    categorical block of a mixed feature table.
    """

    requires_fit = False
    supported_stypes = frozenset({Stype.categorical})

    def _transform(self, table: TableTensor) -> TableTensor:
        _check_categorical_codes(table)

        data = table.categorical.as_tensor().clone()
        categories: list[Tensor] = []
        for index, category in enumerate(table.categorical.categories):
            permutation = self._sorted_indices(
                category,
                device=data.device,
            )
            categories.append(self._select_categories(category, permutation))
            if permutation.numel() == 0:
                continue

            inverse = torch.empty_like(permutation)
            inverse[permutation] = torch.arange(
                permutation.numel(),
                dtype=torch.long,
                device=data.device,
            )
            codes = data[..., index]
            remapped = inverse[codes.clamp_min(0).to(torch.long)]
            data[..., index] = torch.where(
                codes >= 0,
                remapped.to(codes.dtype),
                codes,
            )

        categorical = CategoricalTensor(data=data, categories=categories)
        return table.replace_blocks(categorical=categorical)

    @staticmethod
    def _sorted_indices(
        category: Tensor,
        *,
        device: torch.device,
    ) -> Tensor:
        if (
            isinstance(category, StringTensor)
            or category.dtype in _HOST_SORTED_DTYPES
        ):
            values = category.tolist()
            return torch.tensor(
                sorted(range(category.numel()), key=values.__getitem__),
                dtype=torch.long,
                device=device,
            )
        return category.to(device=device).argsort()

    @staticmethod
    def _select_categories(category: Tensor, index: Tensor) -> Tensor:
        if category.dtype in _HOST_SORTED_DTYPES:
            values = category.tolist()
            return torch.tensor(
                [values[i] for i in index.tolist()],
                dtype=category.dtype,
                device=category.device,
            )
        return category.index_select(0, index.to(device=category.device))
