from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Literal, cast

import torch
from torch import Tensor

from sdm import CategoricalTensor, Stype
from sdm.processing.ensemble import EnsembleProcessor
from sdm.tensor import EnsembleTable, TableTensor


class ShuffleCategories(EnsembleProcessor):
    """Independently permute the integer codes of categorical columns.

    One permutation per categorical column is drawn when the processor is
    fitted; pass ``generator`` to ``fit()`` to make the draws reproducible.
    Codes and their corresponding category vectors are permuted together so
    decoded values remain unchanged. Negative codes represent missing values
    and are preserved unchanged. Only categorical columns are supported; use
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
        *,
        _ensemble_permutations: Callable[
            ...,
            Sequence[Sequence[Sequence[int]]],
        ]
        | None = None,
    ) -> None:
        super().__init__()
        if method not in {"shift", "random"}:
            raise ValueError("method must be 'shift' or 'random'")
        self.method = method
        self._ensemble_permutations = _ensemble_permutations
        self.register_buffer(
            "permutations",
            torch.empty(0, dtype=torch.long),
        )
        self.register_buffer(
            "offsets",
            torch.zeros(1, dtype=torch.long),
        )
        self.processors = torch.nn.ModuleList()
        self._member_to_processor: tuple[int, ...] = ()
        self._processor_positions: tuple[int, ...] = ()
        self._mapping: tuple[tuple[int, ...], ...] = ()
        self._offset_values: tuple[int, ...] = (0,)

    def _fit(
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        device = table.categorical.device
        random_device = device if generator is None else generator.device
        permutations: list[Tensor] = []
        offsets = [0]
        for category in table.categorical.categories:
            n_classes = category.numel()
            if n_classes <= 1:
                permutation = torch.arange(n_classes, device=random_device)
            elif self.method == "shift":
                offset = torch.randint(
                    n_classes,
                    (1,),
                    generator=generator,
                    device=random_device,
                )
                permutation = (
                    torch.arange(n_classes, device=random_device) - offset
                ) % n_classes
            else:
                permutation = torch.randperm(
                    n_classes,
                    generator=generator,
                    device=random_device,
                )
            permutations.append(permutation)
            offsets.append(offsets[-1] + n_classes)

        self._mapping = tuple(
            tuple(permutation.tolist()) for permutation in permutations
        )
        self.permutations = (
            torch.cat(permutations).to(device)
            if len(permutations) > 0
            else torch.empty(0, dtype=torch.long, device=device)
        )
        self.offsets = torch.tensor(
            offsets,
            dtype=torch.long,
            device=device,
        )
        self._offset_values = tuple(offsets)

    def fit_transform_ensemble(
        self,
        table: EnsembleTable,
        *,
        generator: torch.Generator | None = None,
    ) -> EnsembleTable:
        r"""Fit member category mappings and pack compatible outputs."""
        planned = (
            self._ensemble_permutations(
                table._member_ids,
                tuple(
                    tuple(
                        category.numel()
                        for category in table.representation(
                            position
                        ).categorical.categories
                    )
                    for position in range(table.num_members)
                ),
            )
            if self._ensemble_permutations is not None
            else None
        )
        self.processors = torch.nn.ModuleList()
        representations: list[TableTensor] = []
        positions: list[int] = []
        member_to_processor: list[int] = []
        keys: dict[tuple[object, ...], int] = {}
        for position, _ in enumerate(table._member_ids):
            before = table.representation(position)
            processor = self.__class__(method=self.method)
            if planned is None:
                processor.fit(before, generator=generator)
            else:
                processor._mapping = tuple(
                    tuple(value) for value in planned[position]
                )
                offsets = [0]
                for mapping in processor._mapping:
                    offsets.append(offsets[-1] + len(mapping))
                processor.permutations = torch.tensor(
                    tuple(
                        index
                        for mapping in processor._mapping
                        for index in mapping
                    ),
                    dtype=torch.long,
                    device=before.device,
                )
                processor.offsets = torch.tensor(
                    offsets,
                    dtype=torch.long,
                    device=before.device,
                )
                processor._offset_values = tuple(offsets)
                processor._fitted = True

            key = (
                table._member_locations[position],
                processor._mapping,
            )
            processor_index = keys.get(key)
            if processor_index is None:
                processor_index = len(representations)
                keys[key] = processor_index
                self.processors.append(processor)
                positions.append(position)
                representations.append(processor.transform(before))
            member_to_processor.append(processor_index)

        self._member_to_processor = tuple(member_to_processor)
        self._processor_positions = tuple(positions)
        return EnsembleTable.pack(
            representations=representations,
            member_representation_ids=self._member_to_processor,
            member_ids=table._member_ids,
        )

    def transform_ensemble(self, table: EnsembleTable) -> EnsembleTable:
        r"""Apply the fitted member category mappings."""
        representations = tuple(
            cast(ShuffleCategories, processor).transform(
                table.representation(position)
            )
            for processor, position in zip(
                self.processors,
                self._processor_positions,
            )
        )
        return EnsembleTable.pack(
            representations=representations,
            member_representation_ids=self._member_to_processor,
            member_ids=table._member_ids,
        )

    def _transform(self, table: TableTensor) -> TableTensor:
        offsets = self._offset_values
        code = table.categorical.code.clone()
        categories: list[Tensor] = []
        for index, category in enumerate(table.categorical.categories):
            permutation = self.permutations[
                offsets[index] : offsets[index + 1]
            ]
            codes = code[..., index]
            valid = codes >= 0
            valid_codes = codes[valid].to(torch.long)
            codes[valid] = permutation[valid_codes].to(codes.dtype)
            categories.append(category[permutation.argsort()])

        categorical = CategoricalTensor(
            code=code,
            categories=categories,
        )
        return table.replace_blocks(categorical=categorical)
