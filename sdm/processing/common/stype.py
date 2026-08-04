from collections.abc import Sequence
from typing import Literal, cast

import torch
from torch import Tensor

from sdm import Stype
from sdm.processing.base import Processor
from sdm.processing.ensemble import (
    EnsembleInvertibleMixin,
    EnsembleProcessor,
    EnsembleProcessorAdapter,
)
from sdm.tensor import EnsembleTable, TableTensor


class StypeDispatch(EnsembleProcessor, EnsembleInvertibleMixin):
    r"""Apply separate processor pipelines to columns grouped by semantic type.

    For each configured route, the matching columns are selected into a
    :class:`TableTensor` and passed to that processor. A route may change
    column values, names, count, or order. Route outputs are concatenated in
    semantic type order. With the default passthrough behavior, unconfigured
    semantic types follow in input order. A ``generator`` passed to ``fit()``
    or ``fit_transform()`` is passed on to every route.

    Inverse transform supports routes that preserve their semantic type. Every
    active route must be invertible. Tracking transformed ownership for routes
    that change semantic type or share an output semantic type is deferred.
    Passthrough columns are preserved.
    ``remainder="drop"`` is not invertible.

    Args:
        numerical: Processor or stateless callable route for numerical
            columns. An iterable is normalized to
            :class:`~sdm.processing.Sequential`.
        categorical: Processor or stateless callable route for categorical
            columns. An iterable is normalized to
            :class:`~sdm.processing.Sequential`.
        datetime: Processor or stateless callable route for datetime columns.
            An iterable is normalized to
            :class:`~sdm.processing.Sequential`.
        text: Processor or stateless callable route for text columns. An
            iterable is normalized to :class:`~sdm.processing.Sequential`.
        id: Processor or stateless callable route for identifier columns. An
            iterable is normalized to :class:`~sdm.processing.Sequential`.
        remainder: How to handle non-empty semantic types without a configured
            route. ``"passthrough"`` keeps them unchanged and is the default,
            ``"drop"`` removes them, and ``"error"`` raises.
    """

    supported_stypes = frozenset(Stype)

    def __init__(
        self,
        *,
        numerical: object = None,
        categorical: object = None,
        datetime: object = None,
        id: object = None,
        text: object = None,
        remainder: Literal["passthrough", "drop", "error"] = "passthrough",
    ) -> None:
        super().__init__()
        self.processors = torch.nn.ModuleDict()
        for stype, processor in (
            (Stype.numerical, numerical),
            (Stype.categorical, categorical),
            (Stype.datetime, datetime),
            (Stype.text, text),
            (Stype.id, id),
        ):
            if processor is None:
                continue
            self.processors[stype.value] = Processor.as_processor(processor)

        self.remainder = remainder
        self.requires_fit = any(
            processor.requires_fit for processor in self.processors.values()
        )
        self._active_routes: tuple[str, ...] | None = None

    @staticmethod
    def _concatenate_ensemble_tables(
        ensemble_tables: Sequence[EnsembleTable],
        input_table: EnsembleTable,
    ) -> EnsembleTable:
        if len(ensemble_tables) == 0:
            return input_table._replace_groups(
                [group.select_columns(()) for group in input_table]
            )

        if any(
            table.num_members != input_table.num_members
            for table in ensemble_tables
        ):
            raise ValueError(
                "StypeDispatch routes must preserve ensemble member count"
            )

        same_storage_layout = True
        for ensemble_table in ensemble_tables:
            for member_id in range(input_table.num_members):
                member_location = ensemble_table._member_location(member_id)
                input_location = input_table._member_location(member_id)
                if member_location != input_location:
                    same_storage_layout = False
                    break
            if not same_storage_layout:
                break

        if same_storage_layout:
            return input_table._replace_groups(
                [
                    cast(
                        TableTensor,
                        torch.cat(cast(list[Tensor], groups), dim=-1),
                    )
                    for groups in zip(
                        *(tuple(table) for table in ensemble_tables),
                        strict=True,
                    )
                ]
            )

        unique_concatenated_tables: list[TableTensor] = []
        table_id_by_location_combination: dict[
            tuple[tuple[int, int], ...], int
        ] = {}
        member_table_ids = []
        for member_id in range(input_table.num_members):
            location = tuple(
                ensemble_table._member_location(member_id)
                for ensemble_table in ensemble_tables
            )
            if location not in table_id_by_location_combination:
                table_id_by_location_combination[location] = len(
                    unique_concatenated_tables
                )
                unique_concatenated_tables.append(
                    cast(
                        TableTensor,
                        torch.cat(
                            cast(
                                list[Tensor],
                                [
                                    ensemble_table.table(member_id)
                                    for ensemble_table in ensemble_tables
                                ],
                            ),
                            dim=-1,
                        ),
                    )
                )
            member_table_ids.append(table_id_by_location_combination[location])

        return EnsembleTable.from_tables(
            tables=unique_concatenated_tables,
            member_table_ids=member_table_ids,
        )

    def _check_remainder(self, remainder_stypes: list[Stype]) -> None:
        if self.remainder != "error" or len(remainder_stypes) == 0:
            return

        names = ", ".join(f"{stype.value!r}" for stype in remainder_stypes)
        raise ValueError(
            f"Found non-empty input columns for semantic types {names}, but "
            f"{self.__class__.__name__!r} has no route for them. Configure "
            "a processor for each semantic type or set "
            "remainder='passthrough' or remainder='drop'."
        )

    def _fit(
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        remainder_stypes = [
            stype
            for stype, columns in table.columns.items()
            if stype.value not in self.processors and len(columns) > 0
        ]
        self._check_remainder(remainder_stypes)
        for stype, processor in self.processors.items():
            processor = cast(Processor, processor)
            route_input = table.select_stypes(stype)
            if route_input.size(-1) == 0:
                continue
            processor.fit(route_input, generator=generator)

    def _fit_transform(
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> TableTensor:
        remainder_stypes = [
            stype
            for stype, columns in table.columns.items()
            if stype.value not in self.processors and len(columns) > 0
        ]
        self._check_remainder(remainder_stypes)

        outputs: list[TableTensor] = []
        for stype, processor in self.processors.items():
            processor = cast(Processor, processor)
            route_input = table.select_stypes(stype)
            if route_input.size(-1) == 0:
                continue
            out = processor.fit_transform(route_input, generator=generator)
            outputs.append(out)
        if self.remainder == "passthrough":
            outputs.extend(
                table.select_stypes(stype) for stype in remainder_stypes
            )
        if len(outputs) == 0:
            return table.select_columns(())
        return cast(
            TableTensor,
            torch.cat(cast(list[Tensor], outputs), dim=-1),
        )

    def _transform(self, table: TableTensor) -> TableTensor:
        remainder_stypes = [
            stype
            for stype, columns in table.columns.items()
            if stype.value not in self.processors and len(columns) > 0
        ]
        self._check_remainder(remainder_stypes)

        outputs: list[TableTensor] = []
        for stype, processor in self.processors.items():
            processor = cast(Processor, processor)
            route_input = table.select_stypes(stype)
            if route_input.size(-1) == 0:
                continue
            outputs.append(processor.transform(route_input))
        if self.remainder == "passthrough":
            outputs.extend(
                table.select_stypes(stype) for stype in remainder_stypes
            )
        if len(outputs) == 0:
            return table.select_columns(())
        return cast(
            TableTensor,
            torch.cat(cast(list[Tensor], outputs), dim=-1),
        )

    def _fit_transform_ensemble(
        self,
        table: EnsembleTable,
        *,
        generator: torch.Generator | None = None,
    ) -> EnsembleTable:
        for stype, processor in tuple(self.processors.items()):
            self.processors[stype] = EnsembleProcessorAdapter.adapt(
                cast(Processor, processor)
            )

        self._active_routes = tuple(
            stype
            for stype in self.processors
            if any(len(group.columns[Stype(stype)]) > 0 for group in table)
        )
        route_inputs = {
            stype: table._replace_groups(
                [group.select_stypes(stype) for group in table]
            )
            for stype in self._active_routes
        }
        if self.remainder == "error":
            self._remainder_ensemble(table)
        ensemble_tables = [
            cast(
                EnsembleProcessor,
                self.processors[stype],
            ).fit_transform_ensemble(
                route_inputs[stype],
                generator=generator,
            )
            for stype in self._active_routes
        ]
        if self.remainder == "passthrough":
            ensemble_tables.append(self._remainder_ensemble(table))
        return self._concatenate_ensemble_tables(ensemble_tables, table)

    def _transform_ensemble(self, table: EnsembleTable) -> EnsembleTable:
        for stype, processor in tuple(self.processors.items()):
            self.processors[stype] = EnsembleProcessorAdapter.adapt(
                cast(Processor, processor)
            )

        active_routes = self._active_routes
        if not self.requires_fit or active_routes is None:
            active_routes = tuple(
                stype
                for stype in self.processors
                if any(len(group.columns[Stype(stype)]) > 0 for group in table)
            )
            self._active_routes = active_routes
        ensemble_tables = [
            cast(
                EnsembleProcessor,
                self.processors[stype],
            ).transform_ensemble(
                table._replace_groups(
                    [group.select_stypes(stype) for group in table]
                )
            )
            for stype in active_routes
        ]
        if self.remainder == "passthrough":
            ensemble_tables.append(self._remainder_ensemble(table))
        elif self.remainder == "error":
            self._remainder_ensemble(table)
        return self._concatenate_ensemble_tables(ensemble_tables, table)

    def _inverse_transform_ensemble(
        self,
        table: EnsembleTable,
    ) -> EnsembleTable:
        """Invert every active semantic-type route."""
        if self.remainder == "drop":
            raise ValueError(
                "'StypeDispatch' with remainder='drop' is not invertible"
            )

        if self._active_routes is None:
            raise RuntimeError(
                "'StypeDispatch' has no ensemble routing state; call "
                "'fit_transform_ensemble' or 'transform_ensemble' before "
                "'inverse_transform_ensemble'."
            )

        ensemble_tables = []
        for stype in self._active_routes:
            processor = cast(EnsembleProcessor, self.processors[stype])
            inverse = getattr(processor, "inverse_transform_ensemble", None)
            if not callable(inverse):
                raise TypeError(
                    f"Route {stype!r} uses non-invertible processor "
                    f"{processor.__class__.__name__!r}"
                )
            route_input = table._replace_groups(
                [group.select_stypes(stype) for group in table]
            )
            ensemble_tables.append(inverse(route_input))

        ensemble_tables.append(self._remainder_ensemble(table))
        return self._concatenate_ensemble_tables(ensemble_tables, table)

    def _remainder_ensemble(self, table: EnsembleTable) -> EnsembleTable:
        configured = frozenset(self.processors)
        outputs = []
        for group in table:
            remainder = [
                stype
                for stype, columns in group.columns.items()
                if stype.value not in configured and len(columns) > 0
            ]
            self._check_remainder(remainder)
            if len(remainder) == 0:
                outputs.append(group.select_columns(()))
            else:
                outputs.append(group.select_stypes(remainder))
        return table._replace_groups(outputs)

    def _inverse_transform(self, table: TableTensor) -> TableTensor:
        if self.remainder == "drop":
            raise ValueError(
                "'StypeDispatch' with remainder='drop' is not invertible"
            )

        outputs: list[TableTensor] = []
        # TODO: Track transformed route ownership before supporting routes that
        # change stype or share an output stype.
        for stype, processor in self.processors.items():
            processor = cast(Processor, processor)
            route_input = table.select_stypes(stype)
            if route_input.size(-1) == 0:
                continue
            inverse = getattr(processor, "inverse_transform", None)
            if not callable(inverse):
                raise TypeError(
                    f"Route {stype!r} uses non-invertible processor "
                    f"{processor.__class__.__name__!r}"
                )
            outputs.append(inverse(route_input))

        remainder_stypes = [
            stype
            for stype, columns in table.columns.items()
            if stype.value not in self.processors and len(columns) > 0
        ]
        self._check_remainder(remainder_stypes)
        outputs.extend(
            table.select_stypes(stype) for stype in remainder_stypes
        )
        if len(outputs) == 0:
            return table.select_columns(())

        return cast(
            TableTensor,
            torch.cat(cast(list[Tensor], outputs), dim=-1),
        )

    def __repr__(self, *, indent: int = 0) -> str:
        if len(self.processors) == 0:
            return super().__repr__(indent=indent)
        reprs = []
        for stype, processor in self.processors.items():
            processor = cast(Processor, processor)
            processor_repr = processor.__repr__(indent=indent + 2)
            processor_repr = processor_repr[indent + 2 :]
            reprs.append(f"{' ' * (indent + 2)}{stype}: {processor_repr}")
        return (
            f"{' ' * indent}{self.__class__.__name__}(\n"
            + ",\n".join(reprs)
            + f",\n{' ' * indent})"
        )
