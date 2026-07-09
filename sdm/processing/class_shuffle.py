from typing import Any, Literal

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
    preserved unchanged. Only categorical columns are supported; use
    :class:`~sdm.processing.StypeDispatch` to apply this processor to the
    categorical block of a mixed feature table.

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
        expected_columns = len(offsets) - 1
        actual_columns = input.categorical.size(-1)
        if actual_columns != expected_columns:
            raise ValueError(
                "Expected the categorical block to match the fitted column "
                f"count (got {actual_columns} and {expected_columns})"
            )

        data = input.categorical.as_tensor().clone()
        categories: list[Tensor] = []
        for index, category in enumerate(input.categorical.categories):
            expected_classes = offsets[index + 1] - offsets[index]
            if category.numel() != expected_classes:
                column = input.columns[Stype.categorical][index]
                raise ValueError(
                    f"Expected categorical column '{column}' to match the "
                    f"fitted category count (got {category.numel()} and "
                    f"{expected_classes})"
                )

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

    def get_extra_state(self) -> bool:  # noqa: D102
        return self._fitted

    def set_extra_state(self, state: bool) -> None:  # noqa: D102
        self._fitted = state

    def _load_from_state_dict(
        self,
        state_dict: dict[str, Any],
        prefix: str,
        local_metadata: dict[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        for name in ("permutations", "offsets"):
            state = state_dict.get(f"{prefix}{name}")
            if isinstance(state, Tensor):
                buffer = getattr(self, name)
                buffer.resize_(state.size())

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
