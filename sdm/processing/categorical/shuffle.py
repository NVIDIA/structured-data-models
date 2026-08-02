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
        self._member_processor_ids: tuple[int, ...] = ()

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

    def _fit_transform(
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> TableTensor:
        self._fit(table, generator=generator)
        return self._transform(table)

    def _fit_transform_ensemble(
        self,
        table: EnsembleTable,
        *,
        generator: torch.Generator | None = None,
    ) -> EnsembleTable:
        self.processors = torch.nn.ModuleList()
        representations = []
        member_processor_ids = []
        fitted: dict[tuple[tuple[int, int], tuple[int, ...]], int] = {}

        for member_id in range(table.num_members):
            processor = self.__class__(method=self.method)
            transformed = processor.fit_transform(
                table.representation(member_id),
                generator=generator,
            )
            key = (
                table._member_locations[member_id],
                tuple(processor.permutations.tolist()),
            )
            processor_id = fitted.get(key)
            if processor_id is None:
                processor_id = len(representations)
                fitted[key] = processor_id
                self.processors.append(processor)
                representations.append(transformed)
            member_processor_ids.append(processor_id)

        self._member_processor_ids = tuple(member_processor_ids)
        return EnsembleTable.from_representations(
            representations,
            self._member_processor_ids,
        )

    def _transform_ensemble(self, table: EnsembleTable) -> EnsembleTable:
        if len(self._member_processor_ids) != table.num_members:
            raise RuntimeError(
                "ShuffleCategories must be fitted with the same number of "
                "ensemble members before transform."
            )

        representations = []
        member_representation_ids = []
        transformed: dict[tuple[tuple[int, int], int], int] = {}
        for member_id, processor_id in enumerate(self._member_processor_ids):
            key = (table._member_locations[member_id], processor_id)
            representation_id = transformed.get(key)
            if representation_id is None:
                processor = cast(
                    ShuffleCategories,
                    self.processors[processor_id],
                )
                representation_id = len(representations)
                transformed[key] = representation_id
                representations.append(
                    processor.transform(table.representation(member_id))
                )
            member_representation_ids.append(representation_id)

        return EnsembleTable.from_representations(
            representations,
            member_representation_ids,
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
