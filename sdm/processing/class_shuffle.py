from typing import Literal

import torch
from torch import Tensor

from sdm import CategoricalTensor, Stype
from sdm.processing.base import Processor
from sdm.tensor import TableTensor


class ClassShuffle(Processor):
    """Permute the integer codes of a categorical target column.

    The class permutation is drawn from the global CPU generator when the
    processor is fitted; seed with :func:`torch.manual_seed` to make it
    reproducible. Codes and the category vector are permuted together so the
    decoded labels remain unchanged. Negative codes represent missing values
    and are preserved unchanged.

    Args:
        method: Permutation strategy. ``"shift"`` cyclically shifts the
            labels by a drawn offset, and ``"random"`` remaps the labels
            with a drawn permutation.
    """

    supported_stypes = frozenset({Stype.categorical})

    def __init__(
        self,
        method: Literal["shift", "random"] = "shift",
    ) -> None:
        super().__init__()
        self.method = method
        self.register_buffer(
            "permutation",
            torch.empty(0, dtype=torch.long),
        )

    def _fit(self, input: TableTensor) -> None:
        self._check_single_column(input)
        category = input.categorical.categories[0]
        device = input.categorical.device
        n_classes = category.numel()
        if n_classes <= 1:
            self.permutation = torch.arange(n_classes, device=device)
        elif self.method == "shift":
            offset = int(torch.randint(n_classes, (1,)).item())
            self.permutation = (
                torch.arange(n_classes, device=device) - offset
            ) % n_classes
        else:
            self.permutation = torch.randperm(n_classes).to(device=device)

    def _transform(self, input: TableTensor) -> TableTensor:
        self._check_single_column(input)
        category = input.categorical.categories[0]
        if category.numel() != self.permutation.numel():
            raise ValueError(
                "Expected the categorical target to match the fitted class "
                f"count (got {category.numel()} and "
                f"{self.permutation.numel()})"
            )

        categorical = CategoricalTensor(
            data=self._map_codes(
                input.categorical.as_tensor(),
                self.permutation,
            ),
            categories=(category[self.permutation.argsort()],),
        )
        return input.replace_blocks(categorical=categorical)

    def _map_codes(self, input: Tensor, permutation: Tensor) -> Tensor:
        valid = input >= 0
        if not valid.any():
            return input

        codes = input[valid].to(torch.long)
        if codes.max() >= permutation.numel():
            raise ValueError(
                "Expected category codes to be less than the fitted class "
                f"count (got max code {int(codes.max().item())} and "
                f"{permutation.numel()})"
            )

        output = input.clone()
        output[valid] = permutation[codes].to(input.dtype)
        return output

    def _check_single_column(self, input: TableTensor) -> None:
        if input.categorical.size(-1) != 1:
            raise ValueError(
                "Expected exactly one categorical target column "
                f"(got {input.categorical.size(-1)})"
            )
