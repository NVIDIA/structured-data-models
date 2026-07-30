from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, cast

import torch
from torch import Tensor

from sdm import CategoricalTensor, Stype
from sdm.processing.ensemble import (
    EnsembleFitContext,
    EnsembleProcessor,
    EnsembleTable,
)
from sdm.tensor import TableTensor


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
        self.processors = torch.nn.ModuleList()
        self._member_to_processor: tuple[int, ...] = ()
        self._processor_positions: tuple[int, ...] = ()

    def _fit(
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        device = table.categorical.device
        permutations: list[Tensor] = []
        offsets = [0]
        for category in table.categorical.categories:
            n_classes = category.numel()
            if n_classes <= 1:
                permutation = torch.arange(n_classes, device=device)
            elif self.method == "shift":
                offset = torch.randint(
                    n_classes,
                    (1,),
                    generator=generator,
                    device=device,
                )
                permutation = (
                    torch.arange(n_classes, device=device) - offset
                ) % n_classes
            else:
                permutation = torch.randperm(
                    n_classes,
                    generator=generator,
                    device=device,
                )
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

    def _planned_permutations(
        self,
        table: EnsembleTable,
        context: EnsembleFitContext,
    ) -> tuple[tuple[tuple[int, ...], ...], ...] | None:
        if context.planner is None:
            return None
        return context.planner.category_permutations(
            member_ids=context.member_ids,
            category_counts=tuple(
                tuple(
                    category.numel()
                    for category in table[position].categorical.categories
                )
                for position in range(table.num_members)
            ),
            table_scope=context.table_scope,
            processor_path=context.processor_path,
        )

    @staticmethod
    def _set_permutations(
        processor: ShuffleCategories,
        table: TableTensor,
        permutations: Sequence[Sequence[int]],
    ) -> None:
        permutations = tuple(tuple(value) for value in permutations)
        counts = tuple(
            category.numel() for category in table.categorical.categories
        )
        if len(permutations) != len(counts) or any(
            sorted(permutation) != list(range(count))
            for permutation, count in zip(permutations, counts)
        ):
            raise ValueError(
                "The ensemble plan returned an invalid category permutation."
            )
        offsets = [0]
        for count in counts:
            offsets.append(offsets[-1] + count)
        processor.permutations = torch.tensor(
            tuple(index for value in permutations for index in value),
            dtype=torch.long,
            device=table.device,
        )
        processor.offsets = torch.tensor(
            offsets,
            dtype=torch.long,
            device=table.device,
        )
        processor._fitted = True

    def fit_transform_ensemble(
        self,
        table: EnsembleTable,
        *,
        context: EnsembleFitContext,
    ) -> EnsembleTable:
        r"""Fit member category mappings and pack compatible outputs."""
        planned = self._planned_permutations(table, context)
        if planned is not None and len(planned) != table.num_members:
            raise ValueError(
                "The ensemble plan must return one mapping per member."
            )

        self.processors = torch.nn.ModuleList()
        variants: list[TableTensor] = []
        positions: list[int] = []
        member_to_processor: list[int] = []
        keys: dict[tuple[object, ...], int] = {}
        for position, member_id in enumerate(context.member_ids):
            before = table[position]
            processor = self.__class__(method=self.method)
            if planned is None:
                processor.fit(
                    before,
                    generator=context.generator_for(
                        member_id,
                        device=before.device,
                    ),
                )
                mapping_key: tuple[object, ...] = ("member", member_id)
            else:
                self._set_permutations(
                    processor,
                    before,
                    planned[position],
                )
                mapping_key = tuple(
                    tuple(permutation) for permutation in planned[position]
                )

            key = (table.member_to_variant[position], mapping_key)
            processor_index = keys.get(key)
            if processor_index is None:
                processor_index = len(variants)
                keys[key] = processor_index
                self.processors.append(processor)
                positions.append(position)
                variants.append(processor.transform(before))
            member_to_processor.append(processor_index)

        self._member_to_processor = tuple(member_to_processor)
        self._processor_positions = tuple(positions)
        return EnsembleTable.pack(
            variants=variants,
            member_to_input_variant=self._member_to_processor,
        )

    def transform_ensemble(self, table: EnsembleTable) -> EnsembleTable:
        r"""Apply the fitted member category mappings."""
        variants = tuple(
            cast(ShuffleCategories, processor).transform(table[position])
            for processor, position in zip(
                self.processors,
                self._processor_positions,
            )
        )
        return EnsembleTable.pack(
            variants=variants,
            member_to_input_variant=self._member_to_processor,
        )

    def _transform(self, table: TableTensor) -> TableTensor:
        offsets = self.offsets.tolist()
        code = table.categorical.code.clone()
        categories: list[Tensor] = []
        for index, category in enumerate(table.categorical.categories):
            permutation = self.permutations[
                offsets[index] : offsets[index + 1]
            ]
            codes = code[..., index]
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
            code=code,
            categories=categories,
        )
        return table.replace_blocks(categorical=categorical)
