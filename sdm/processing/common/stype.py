from collections.abc import Sequence
from typing import Literal, cast

import torch
from torch import Tensor

from sdm import Stype
from sdm.processing.base import InvertibleMixin, Processor
from sdm.processing.ensemble import (
    EnsembleProcessor,
    EnsembleProcessorAdapter,
)
from sdm.tensor import EnsembleTable, TableTensor


def _combine_ensemble_parts(
    parts: Sequence[EnsembleTable],
    source: EnsembleTable,
) -> EnsembleTable:
    if len(parts) == 0:
        return source._replace_packed_representations(
            [
                packed.select_columns(())
                for packed in source.iter_packed_representations()
            ]
        )

    if all(
        all(
            part.member_location(member_id)
            == source.member_location(member_id)
            for member_id in range(source.num_members)
        )
        for part in parts
    ):
        return source._replace_packed_representations(
            [
                cast(
                    TableTensor,
                    torch.cat(cast(list[Tensor], packed_parts), dim=-1),
                )
                for packed_parts in zip(
                    *(
                        tuple(part.iter_packed_representations())
                        for part in parts
                    ),
                    strict=True,
                )
            ]
        )

    representations: list[TableTensor] = []
    locations: dict[tuple[tuple[int, int], ...], int] = {}
    member_representation_ids = []
    for member_id in range(source.num_members):
        location = tuple(part.member_location(member_id) for part in parts)
        if location not in locations:
            locations[location] = len(representations)
            representations.append(
                cast(
                    TableTensor,
                    torch.cat(
                        cast(
                            list[Tensor],
                            [part.representation(member_id) for part in parts],
                        ),
                        dim=-1,
                    ),
                )
            )
        member_representation_ids.append(locations[location])

    return EnsembleTable.from_representations(
        representations=representations,
        member_representation_ids=member_representation_ids,
    )


class StypeDispatch(EnsembleProcessor, InvertibleMixin):
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
        self._active_routes: tuple[str, ...] = ()

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
            if any(
                len(packed.columns[Stype(stype)]) > 0
                for packed in table.iter_packed_representations()
            )
        )
        route_inputs = {
            stype: table._replace_packed_representations(
                [
                    packed.select_stypes(stype)
                    for packed in table.iter_packed_representations()
                ]
            )
            for stype in self._active_routes
        }
        parts = [
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
            parts.append(self._remainder_ensemble(table))
        elif self.remainder == "error":
            self._remainder_ensemble(table)
        return _combine_ensemble_parts(parts, table)

    def _transform_ensemble(self, table: EnsembleTable) -> EnsembleTable:
        for stype, processor in tuple(self.processors.items()):
            self.processors[stype] = EnsembleProcessorAdapter.adapt(
                cast(Processor, processor)
            )

        active_routes = self._active_routes
        if not self.requires_fit:
            active_routes = tuple(
                stype
                for stype in self.processors
                if any(
                    len(packed.columns[Stype(stype)]) > 0
                    for packed in table.iter_packed_representations()
                )
            )
            self._active_routes = active_routes
        parts = [
            cast(
                EnsembleProcessor,
                self.processors[stype],
            ).transform_ensemble(
                table._replace_packed_representations(
                    [
                        packed.select_stypes(stype)
                        for packed in table.iter_packed_representations()
                    ]
                )
            )
            for stype in active_routes
        ]
        if self.remainder == "passthrough":
            parts.append(self._remainder_ensemble(table))
        elif self.remainder == "error":
            self._remainder_ensemble(table)
        return _combine_ensemble_parts(parts, table)

    def inverse_transform_ensemble(
        self,
        table: EnsembleTable,
    ) -> EnsembleTable:
        """Invert every active semantic-type route."""
        self._check_is_fitted()
        if self.remainder == "drop":
            raise ValueError(
                "'StypeDispatch' with remainder='drop' is not invertible"
            )

        parts = []
        for stype in self._active_routes:
            processor = cast(EnsembleProcessor, self.processors[stype])
            inverse = getattr(processor, "inverse_transform_ensemble", None)
            if inverse is None:
                raise TypeError(
                    f"Route {stype!r} uses non-invertible processor "
                    f"{processor.__class__.__name__!r}"
                )
            route_input = table._replace_packed_representations(
                [
                    packed.select_stypes(stype)
                    for packed in table.iter_packed_representations()
                ]
            )
            parts.append(inverse(route_input))

        parts.append(self._remainder_ensemble(table))
        return _combine_ensemble_parts(parts, table)

    def _remainder_ensemble(self, table: EnsembleTable) -> EnsembleTable:
        configured = frozenset(self.processors)
        outputs = []
        for packed in table.iter_packed_representations():
            remainder = [
                stype
                for stype, columns in packed.columns.items()
                if stype.value not in configured and len(columns) > 0
            ]
            self._check_remainder(remainder)
            if len(remainder) == 0:
                outputs.append(packed.select_columns(()))
            else:
                outputs.append(packed.select_stypes(remainder))
        return table._replace_packed_representations(outputs)

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
