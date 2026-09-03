from typing import Literal, cast

import torch
from torch import Tensor
from torch.nn import ModuleDict

from sdm import Stype, TableTensor
from sdm.processing import EnsembleProcessor, Processor
from sdm.tensor import EnsembleTable


class ContextQueryDispatch(EnsembleProcessor):
    """Apply separate feature processors to context and query tables.

    :class:`ContextQueryDispatch` is resolved only during model execution, and
    context and query processors must preserve the same feature schema.

    Args:
        context: Processor used for the context table.
        query: Processor used for the query. Query processors need to be
            stateless.
    """

    requires_fit: bool = True

    def __init__(
        self,
        *,
        context: object = None,
        query: object = None,
    ) -> None:
        super().__init__()
        self.processors: ModuleDict[EnsembleProcessor] = ModuleDict()
        for route, processor in (("context", context), ("query", query)):
            if processor is None:
                continue
            self.processors[route] = EnsembleProcessor.as_processor(processor)

        self._route: Literal["context", "query"] | None = None
        self._locations: tuple[tuple[int, int], ...] | None = None

    @property
    def handles_stypes(self) -> frozenset[Stype]:
        r""":meta private:"""  # noqa: D415
        return frozenset(
            stype
            for processor in self.processors.values()
            for stype in processor.handles_stypes
        )

    def _fit_ensemble(
        self,
        ensemble_table: EnsembleTable,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        self._fit_transform_ensemble(ensemble_table, generator=generator)

    def _fit_transform_ensemble(
        self,
        ensemble_table: EnsembleTable,
        *,
        generator: torch.Generator | None = None,
    ) -> EnsembleTable:
        if self._route is None:
            raise RuntimeError(
                f"{self.__class__.__name__!r} has no resolved context route; "
                "use it in a 'Recipe' through model execution"
            )

        if self._route == "query":
            raise RuntimeError(
                f"{self.__class__.__name__!r} cannot fit query tables"
            )

        if self._route in self.processors:
            processor = self.processors[self._route]
            ensemble_table = processor.fit_transform_ensemble(
                ensemble_table,
                generator=generator,
            )

        self._locations = ensemble_table._locations

        return ensemble_table

    def _transform_ensemble(
        self,
        ensemble_table: EnsembleTable,
    ) -> EnsembleTable:
        if self._route is None:
            raise RuntimeError(
                f"{self.__class__.__name__!r} has no resolved context route; "
                "use it in a 'Recipe' through model execution"
            )

        if self._route in self.processors:
            processor = self.processors[self._route]
            ensemble_table = processor.transform_ensemble(ensemble_table)

        if (  # Regroup query table to match fitted context ensemble layout:
            self._route == "query"
            and ensemble_table._locations != self._locations
        ):
            assert self._locations is not None
            num_groups = max(group_id for group_id, _ in self._locations) + 1
            groups: list[list[TableTensor | None]] = [
                [] for _ in range(num_groups)
            ]
            for group_id, _ in self._locations:
                groups[group_id].append(None)
            for i, (group_id, position) in enumerate(self._locations):
                groups[group_id][position] = ensemble_table.table(i)

            ensemble_table = EnsembleTable._from_groups(
                groups=[
                    cast(TableTensor, group[0].unsqueeze(0))
                    if len(group) == 1
                    else cast(TableTensor, torch.stack(group, dim=0))
                    for group in cast(list[list[Tensor]], groups)
                ],
                locations=self._locations,
            )

        return ensemble_table

    def get_extra_state(
        self,
    ) -> tuple[str | None, tuple[tuple[int, int], ...] | None]:
        r""":meta private:"""  # noqa: D415
        return (self._route, self._locations)

    def set_extra_state(
        self,
        state: tuple[str | None, tuple[tuple[int, int], ...] | None],
    ) -> None:
        r""":meta private:"""  # noqa: D415
        self._route = cast(Literal["context", "query"] | None, state[0])
        self._locations = state[1]

    def __repr__(self, *, indent: int = 0) -> str:
        if len(self.processors) == 0:
            return super().__repr__(indent=indent)

        reprs = []
        for route, processor in self.processors.items():
            processor = cast(Processor, processor)
            processor_repr = processor.__repr__(indent=indent + 2)
            processor_repr = processor_repr[indent + 2 :]
            reprs.append(f"{' ' * (indent + 2)}{route}={processor_repr}")
        return (
            f"{' ' * indent}{self.__class__.__name__}(\n"
            + ",\n".join(reprs)
            + f",\n{' ' * indent})"
        )
