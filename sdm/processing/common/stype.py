from collections.abc import Sequence
from typing import Literal, cast

import torch
from torch import Tensor

from sdm import Stype
from sdm.processing.base import InvertibleMixin, Processor
from sdm.processing.ensemble import (
    EnsembleFitContext,
    EnsembleProcessor,
    EnsembleTable,
    as_ensemble_processor,
)
from sdm.tensor import TableTensor


def _combine_parts(
    parts: Sequence[EnsembleTable],
    *,
    empty_source: EnsembleTable,
) -> EnsembleTable:
    if len(parts) == 0:
        return empty_source.with_groups(
            tuple(group.select_columns(()) for group in empty_source.groups)
        )

    variants: list[TableTensor] = []
    keys: dict[tuple[tuple[int, int], ...], int] = {}
    member_to_variant: list[int] = []
    for member in range(parts[0].num_members):
        key = tuple(part.member_to_variant[member] for part in parts)
        if key not in keys:
            keys[key] = len(variants)
            tables = [part[member] for part in parts]
            variants.append(
                cast(
                    TableTensor,
                    torch.cat(cast(list[Tensor], tables), dim=-1),
                )
                if len(tables) > 1
                else tables[0]
            )
        member_to_variant.append(keys[key])
    return EnsembleTable.pack(
        variants=variants,
        member_to_input_variant=member_to_variant,
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

    def _route_input(
        self,
        table: EnsembleTable,
        stype: str,
    ) -> EnsembleTable:
        return table.with_groups(
            tuple(group.select_stypes(stype) for group in table.groups)
        )

    def _remainder_input(self, table: EnsembleTable) -> EnsembleTable:
        configured = frozenset(self.processors)

        def select(variant: TableTensor) -> TableTensor:
            remainder = tuple(
                stype
                for stype, columns in variant.columns.items()
                if stype.value not in configured and len(columns) > 0
            )
            self._check_remainder(list(remainder))
            if len(remainder) == 0:
                return variant.select_columns(())
            return variant.select_stypes(remainder)

        return table.with_groups(
            tuple(select(group) for group in table.groups)
        )

    def fit_transform_ensemble(
        self,
        table: EnsembleTable,
        *,
        context: EnsembleFitContext,
    ) -> EnsembleTable:
        r"""Fit active semantic-type routes and combine their outputs."""
        for stype, route in tuple(self.processors.items()):
            self.processors[stype] = as_ensemble_processor(
                cast(Processor, route)
            )

        route_inputs = {
            stype: self._route_input(table, stype) for stype in self.processors
        }
        self._active_routes = tuple(
            stype
            for stype, route_input in route_inputs.items()
            if any(group.size(-1) > 0 for group in route_input.groups)
        )

        parts = [
            cast(
                EnsembleProcessor,
                self.processors[stype],
            ).fit_transform_ensemble(
                route_inputs[stype],
                context=context.child(stype),
            )
            for stype in self._active_routes
        ]
        if self.remainder == "passthrough":
            parts.append(self._remainder_input(table))
        elif self.remainder == "error":
            self._remainder_input(table)
        return _combine_parts(parts, empty_source=table)

    def transform_ensemble(self, table: EnsembleTable) -> EnsembleTable:
        r"""Transform active routes and combine their outputs."""
        parts = [
            cast(
                EnsembleProcessor,
                self.processors[stype],
            ).transform_ensemble(self._route_input(table, stype))
            for stype in self._active_routes
        ]
        if self.remainder == "passthrough":
            parts.append(self._remainder_input(table))
        elif self.remainder == "error":
            self._remainder_input(table)
        return _combine_parts(parts, empty_source=table)

    def inverse_transform_ensemble(
        self,
        table: EnsembleTable,
    ) -> EnsembleTable:
        r"""Invert the single active semantic-type route."""
        if len(self._active_routes) != 1:
            raise NotImplementedError(
                "Ensemble inverse transform requires one active "
                "StypeDispatch route."
            )
        return cast(
            EnsembleProcessor,
            self.processors[self._active_routes[0]],
        ).inverse_transform_ensemble(table)

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
            if not isinstance(processor, InvertibleMixin):
                raise TypeError(
                    f"Route {stype!r} uses non-invertible processor "
                    f"{processor.__class__.__name__!r}"
                )
            outputs.append(processor.inverse_transform(route_input))

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
