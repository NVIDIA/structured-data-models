from collections.abc import Sequence
from typing import cast

import torch

from sdm.processing.ensemble import EnsembleProcessor
from sdm.stype import Stype
from sdm.tensor import EnsembleTable, TableTensor


class PCA(EnsembleProcessor):
    """Project numerical columns onto their principal components.

    The mean and components are fitted on the context table via a singular
    value decomposition of the centered data. The effective dimension is
    capped at the numerical rank of the centered data. Output columns are named
    ``"pca_0"``, ..., ``"pca_{num_components-1}"``.

    Args:
        num_components: Number of principal components to keep.
    """

    supported_stypes = frozenset({Stype.numerical})

    def __init__(self, *, num_components: int) -> None:
        super().__init__()
        if num_components < 1:
            raise ValueError(
                f"'num_components' must be positive (got {num_components})."
            )

        self.num_components = num_components
        self.register_buffer("mean", torch.empty(0))
        self.register_buffer("components", torch.empty(0))
        self.processors = torch.nn.ModuleList()
        self._member_processor_ids: tuple[int, ...] = ()

    def _fit(
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        numerical = table.numerical
        if numerical.numel() == 0:
            raise ValueError("`table` must be non-empty.")

        numerical = numerical.flatten(end_dim=-2)  # [..., F] -> [N, F]
        self.mean = numerical.mean(dim=0)
        # Economy SVD; right-singular vectors are the principal axes.
        _, singular_values, vh = torch.linalg.svd(
            numerical - self.mean,
            full_matrices=False,
        )
        tolerance = (
            singular_values.max()
            * max(numerical.shape)
            * torch.finfo(singular_values.dtype).eps
        )
        rank = int((singular_values > tolerance).sum())
        num_components = min(self.num_components, rank)
        self.components = vh[:num_components].T
        self._columns: dict[str, Sequence[str]] = {
            Stype.numerical: tuple(f"pca_{i}" for i in range(num_components))
        }

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
        processors = torch.nn.ModuleList()
        representations: list[TableTensor] = []
        member_processor_ids: list[int] = []
        fitted: dict[tuple[int, int], int] = {}

        for member_id in range(table.num_members):
            location = table._member_locations[member_id]
            processor_id = fitted.get(location)
            if processor_id is None:
                processor = self.__class__(num_components=self.num_components)
                transformed = processor.fit_transform(
                    table.representation(member_id),
                    generator=generator,
                )
                processor_id = len(processors)
                fitted[location] = processor_id
                processors.append(processor)
                representations.append(transformed)
            member_processor_ids.append(processor_id)

        self.processors = processors
        self._member_processor_ids = tuple(member_processor_ids)
        return EnsembleTable.from_representations(
            representations,
            member_processor_ids,
        )

    def _transform_ensemble(self, table: EnsembleTable) -> EnsembleTable:
        if len(self._member_processor_ids) != table.num_members:
            raise RuntimeError(
                "PCA must be fitted with the same number of ensemble members "
                "before transform."
            )

        representations: list[TableTensor] = []
        member_representation_ids: list[int] = []
        transformed: dict[tuple[tuple[int, int], int], int] = {}
        for member_id, processor_id in enumerate(self._member_processor_ids):
            key = (table._member_locations[member_id], processor_id)
            representation_id = transformed.get(key)
            if representation_id is None:
                processor = cast(PCA, self.processors[processor_id])
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
        if table.numerical.size(-1) != self.mean.size(0):
            raise ValueError(
                f"Expected 'table' to have {self.mean.size(0)} numerical "
                f"columns, matching the table used to fit 'PCA' (got "
                f"{table.numerical.size(-1)})."
            )

        numerical = table.numerical
        output_shape = (*numerical.shape[:-1], self.components.size(-1))
        numerical = numerical.flatten(end_dim=-2)
        projected = (numerical - self.mean) @ self.components  # [N, C]
        projected = projected.reshape(output_shape)  # [N, C] -> [..., C]
        return TableTensor(
            columns=self._columns,
            numerical=projected,
        )
