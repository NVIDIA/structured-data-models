from typing import Literal

import torch
from torch import Tensor

from sdm import CategoricalTensor, Stype
from sdm.processing.base import Processor
from sdm.tensor import TableTensor


class ClassShuffle(Processor):
    """Independently permute the integer codes of categorical columns.

    One permutation per categorical column is drawn from the global CPU
    generator when the processor is fitted; seed with
    :func:`torch.manual_seed` to make the draws reproducible. Codes and their
    corresponding category vectors are permuted together so decoded values
    remain unchanged. Negative codes represent missing values and are
    preserved unchanged. Non-categorical blocks pass through unchanged.

    Args:
        method: Permutation strategy. ``"shift"`` cyclically shifts the
            codes by a drawn offset, and ``"random"`` remaps the codes with
            a drawn permutation.
    """

    supported_stypes = frozenset(Stype)

    def __init__(
        self,
        method: Literal["shift", "random"] = "shift",
    ) -> None:
        super().__init__()
        self.method = method
        self.register_buffer(
            "permutations",
            torch.empty(0, dtype=torch.long),
        )
        self.register_buffer(
            "offsets",
            torch.zeros(1, dtype=torch.long),
        )

    def _fit(self, input: TableTensor) -> None:
        device = input.categorical.device
        permutations: list[Tensor] = []
        offsets = [0]
        for category in input.categorical.categories:
            permutation = self._draw_permutation(category.numel(), device)
            permutations.append(permutation)
            offsets.append(offsets[-1] + category.numel())

        self.permutations = (
            torch.cat(permutations)
            if len(permutations) > 0
            else torch.empty(0, dtype=torch.long, device=device)
        )
        self.offsets = torch.tensor(
            offsets,
            dtype=torch.long,
            device=device,
        )

    def _draw_permutation(
        self,
        n_classes: int,
        device: torch.device,
    ) -> Tensor:
        if n_classes <= 1:
            return torch.arange(n_classes, device=device)
        if self.method == "shift":
            offset = int(torch.randint(n_classes, (1,)).item())
            return (
                torch.arange(n_classes, device=device) - offset
            ) % n_classes
        return torch.randperm(n_classes).to(device=device)

    def _transform(self, input: TableTensor) -> TableTensor:
        self._check_schema(input)
        if input.categorical.size(-1) == 0:
            return input

        data = input.categorical.as_tensor().clone()
        categories: list[Tensor] = []
        offsets = self.offsets.tolist()
        for index, category in enumerate(input.categorical.categories):
            permutation = self.permutations[
                offsets[index] : offsets[index + 1]
            ]
            data[..., index] = self._map_codes(
                data[..., index],
                permutation,
            )
            categories.append(category[permutation.argsort()])

        categorical = CategoricalTensor(
            data=data,
            categories=categories,
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

    def _check_schema(self, input: TableTensor) -> None:
        expected_columns = self.offsets.numel() - 1
        actual_columns = input.categorical.size(-1)
        if actual_columns != expected_columns:
            raise ValueError(
                "Expected the categorical block to match the fitted column "
                f"count (got {actual_columns} and {expected_columns})"
            )

        offsets = self.offsets.tolist()
        for index, category in enumerate(input.categorical.categories):
            expected_classes = offsets[index + 1] - offsets[index]
            if category.numel() != expected_classes:
                column = input.columns[Stype.categorical][index]
                raise ValueError(
                    f"Expected categorical column '{column}' to match the "
                    f"fitted category count (got {category.numel()} and "
                    f"{expected_classes})"
                )
