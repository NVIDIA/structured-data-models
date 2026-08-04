from typing import Literal, cast

import torch
from torch import Tensor

from sdm import Stype
from sdm.processing.ensemble import EnsembleInvertibleMixin, EnsembleProcessor
from sdm.tensor import EnsembleTable, TableTensor


class ShuffleColumns(EnsembleProcessor, EnsembleInvertibleMixin):
    """Permute the numerical feature columns.

    The permutation is drawn when the processor is fitted; pass
    ``generator`` to ``fit()`` to make it reproducible. Convert
    non-numerical feature stypes before this step, for example with
    :class:`~sdm.processing.ToNumerical`.

    Args:
        method: Permutation strategy. ``"shift"`` cyclically shifts the
            columns by a drawn offset, and ``"random"`` permutes the columns
            with a drawn permutation.
    """

    supported_stypes = frozenset({Stype.numerical})

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
        self.processors = torch.nn.ModuleList()
        self._member_processor_ids: tuple[int, ...] = ()

    def _fit(
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        n_features = table.numerical.size(-1)
        device = table.numerical.device
        if n_features <= 1:
            self.permutation = torch.arange(n_features, device=device)
        elif self.method == "shift":
            offset = torch.randint(
                n_features,
                (1,),
                generator=generator,
                device=device,
            )
            self.permutation = (
                torch.arange(n_features, device=device) + offset
            ) % n_features
        else:
            self.permutation = torch.randperm(
                n_features,
                generator=generator,
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

    def _fit_ensemble(
        self,
        ensemble_table: EnsembleTable,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        self.processors = torch.nn.ModuleList()
        member_processor_ids = []
        fitted: dict[tuple[tuple[int, int], tuple[int, ...]], int] = {}

        for member_id in range(ensemble_table.num_members):
            processor = self.__class__(method=self.method)
            processor.fit(
                ensemble_table.table(member_id),
                generator=generator,
            )
            key = (
                ensemble_table._locations[member_id],
                tuple(processor.permutation.tolist()),
            )
            processor_id = fitted.get(key)
            if processor_id is None:
                processor_id = len(self.processors)
                fitted[key] = processor_id
                self.processors.append(processor)
            member_processor_ids.append(processor_id)

        self._member_processor_ids = tuple(member_processor_ids)

    def _fit_transform_ensemble(
        self,
        ensemble_table: EnsembleTable,
        *,
        generator: torch.Generator | None = None,
    ) -> EnsembleTable:
        self._fit_ensemble(ensemble_table, generator=generator)
        return self._apply_ensemble(ensemble_table, inverse=False)

    def _transform_ensemble(
        self,
        ensemble_table: EnsembleTable,
    ) -> EnsembleTable:
        return self._apply_ensemble(ensemble_table, inverse=False)

    def _inverse_transform_ensemble(
        self,
        ensemble_table: EnsembleTable,
    ) -> EnsembleTable:
        return self._apply_ensemble(ensemble_table, inverse=True)

    def _apply_ensemble(
        self,
        ensemble_table: EnsembleTable,
        *,
        inverse: bool,
    ) -> EnsembleTable:
        if len(self._member_processor_ids) != ensemble_table.num_members:
            operation = "inverse transform" if inverse else "transform"
            raise RuntimeError(
                "ShuffleColumns must be fitted with the same number of "
                f"ensemble members before {operation}."
            )

        tables = []
        member_table_ids = []
        transformed: dict[tuple[tuple[int, int], int], int] = {}
        for member_id, processor_id in enumerate(self._member_processor_ids):
            key = (ensemble_table._locations[member_id], processor_id)
            table_id = transformed.get(key)
            if table_id is None:
                processor = cast(
                    ShuffleColumns,
                    self.processors[processor_id],
                )
                table_id = len(tables)
                transformed[key] = table_id
                member_table = ensemble_table.table(member_id)
                tables.append(
                    processor.inverse_transform(member_table)
                    if inverse
                    else processor.transform(member_table)
                )
            member_table_ids.append(table_id)

        return EnsembleTable.from_tables(
            tables=tables,
            member_table_ids=member_table_ids,
        )

    def _transform(self, table: TableTensor) -> TableTensor:
        """Reorder the numerical block with the fitted permutation."""
        return self._permute(table, self.permutation)

    def _inverse_transform(self, table: TableTensor) -> TableTensor:
        return self._permute(table, self.permutation.argsort())

    def _permute(
        self,
        table: TableTensor,
        permutation: Tensor,
    ) -> TableTensor:
        if len(self.processors) > 0:
            raise RuntimeError(
                "'ShuffleColumns' was fitted for an ensemble; use the "
                "ensemble transform methods."
            )
        indices = permutation.tolist()
        return table.__class__(
            columns={
                Stype.numerical.value: tuple(
                    table.columns[Stype.numerical][index] for index in indices
                )
            },
            numerical=table.numerical.index_select(-1, permutation),
        )
