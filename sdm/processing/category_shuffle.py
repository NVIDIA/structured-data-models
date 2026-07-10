from typing import Literal

import torch
from torch import Tensor

from sdm import CategoricalTensor, Stype
from sdm.processing.base import InvertibleMixin, Processor
from sdm.tensor import TableTensor


class CategoryShuffle(Processor, InvertibleMixin):
    """Independently permute the integer codes of categorical columns.

    One permutation per categorical column is drawn from the global CPU
    generator when the processor is fitted; seed with
    :func:`torch.manual_seed` to make the draws reproducible. Codes and their
    corresponding category vectors are permuted together so decoded values
    remain unchanged. Negative codes represent missing values and are
    preserved unchanged. Only categorical columns are supported; use
    :class:`~sdm.processing.StypeDispatch` to apply this processor to the
    categorical block of a mixed feature table.

    When fitted on exactly one categorical target,
    :meth:`~sdm.processing.InvertibleMixin.inverse_transform` interprets its
    input as a numerical model-output table containing class scores in
    shuffled-code order. It drops inactive trailing head entries and restores
    the fitted original class order.

    Args:
        method: Permutation strategy. ``"shift"`` cyclically shifts the
            codes by a drawn offset, and ``"random"`` remaps the codes with
            a drawn permutation.
    """

    supported_stypes = frozenset({Stype.categorical})

    def __init__(
        self,
        method: Literal["shift", "random"] = "shift",
    ) -> None:
        super().__init__()
        if method not in {"shift", "random"}:
            raise ValueError("method must be 'shift' or 'random'")
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
            n_classes = category.numel()
            if n_classes <= 1:
                permutation = torch.arange(n_classes, device=device)
            elif self.method == "shift":
                offset = int(torch.randint(n_classes, (1,)).item())
                permutation = (
                    torch.arange(n_classes, device=device) - offset
                ) % n_classes
            else:
                permutation = torch.randperm(n_classes).to(device=device)
            permutations.append(permutation)
            offsets.append(offsets[-1] + n_classes)

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

    def _transform(self, input: TableTensor) -> TableTensor:
        offsets = self.offsets.tolist()
        data = input.categorical.as_tensor().clone()
        categories: list[Tensor] = []
        for index, category in enumerate(input.categorical.categories):
            permutation = self.permutations[
                offsets[index] : offsets[index + 1]
            ]
            codes = data[..., index]
            valid = codes >= 0
            if valid.any():
                valid_codes = codes[valid].to(torch.long)
                max_code = int(valid_codes.max().item())
                if max_code >= permutation.numel():
                    raise ValueError(
                        "Expected category codes to be less than the fitted "
                        f"class count (got max code {max_code} and "
                        f"{permutation.numel()})"
                    )
                codes[valid] = permutation[valid_codes].to(codes.dtype)
            categories.append(category[permutation.argsort()])

        categorical = CategoricalTensor(
            data=data,
            categories=categories,
        )
        return input.replace_blocks(categorical=categorical)

    def _inverse_transform(self, input: TableTensor) -> TableTensor:
        """Restore original class order in a model-output score table.

        Args:
            input: Numerical model output whose last dimension contains class
                scores in shuffled-code order. Arbitrary leading dimensions
                are preserved, and trailing entries beyond the fitted class
                count are ignored.

        Returns:
            Numerical table with one score column per fitted category in
            original class order.

        Raises:
            ValueError: If this processor was fitted on anything other than
                exactly one categorical column.
        """
        n_columns = self.offsets.numel() - 1
        if n_columns != 1:
            raise ValueError(
                f"Expected '{self.__class__.__name__}' to be fitted on "
                f"exactly one categorical column for inverse model-output "
                f"processing (got {n_columns} columns)"
            )

        permutation = self.permutations
        scores = input.numerical[..., : permutation.numel()].index_select(
            dim=-1,
            index=permutation,
        )
        return TableTensor.from_tensor(scores)
