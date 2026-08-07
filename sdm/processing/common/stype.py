from typing import Literal, cast

import torch

from sdm import Stype
from sdm.processing import (
    EnsembleInvertibleMixin,
    EnsembleProcessor,
    EnsembleProcessorAdapter,
    Processor,
)
from sdm.tensor import EnsembleTable


class StypeDispatch(EnsembleProcessor, EnsembleInvertibleMixin):
    r"""Apply separate processor pipelines to columns grouped by semantic type.

    For each configured route, matching columns from a
    :class:`~sdm.tensor.TableTensor` or
    :class:`~sdm.tensor.EnsembleTable` are passed to that processor. Ordinary
    processors learn separate state for each compatible ensemble group, while
    ensemble-aware processors operate on all groups directly. Compatible
    members are processed together and route outputs are concatenated in
    semantic type order while preserving logical member order. A route may
    change column values, names, count, or order. With the default passthrough
    behavior, unconfigured semantic types follow in input order. A
    ``generator`` passed during fitting is passed on to every route.

    Inverse transform supports routes that preserve their semantic type. Every
    active route must be invertible, and routes must not share an output
    semantic type. Passthrough columns are preserved. ``remainder="drop"`` is
    not invertible.

    Args:
        numerical: Processor or stateless callable route for numerical
            columns. A sequence is normalized to
            :class:`~sdm.processing.Sequential`.
        categorical: Processor or stateless callable route for categorical
            columns. A sequence is normalized to
            :class:`~sdm.processing.Sequential`.
        datetime: Processor or stateless callable route for datetime columns.
            A sequence is normalized to
            :class:`~sdm.processing.Sequential`.
        text: Processor or stateless callable route for text columns. A
            sequence is normalized to :class:`~sdm.processing.Sequential`.
        id: Processor or stateless callable route for identifier columns. A
            sequence is normalized to :class:`~sdm.processing.Sequential`.
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
            processor = Processor.as_processor(processor)
            if not isinstance(processor, EnsembleProcessor):
                processor = EnsembleProcessorAdapter(processor)
            self.processors[str(stype)] = processor

        self.remainder = remainder
        self.requires_fit = any(
            processor.requires_fit for processor in self.processors.values()
        )
        self._active_routes: tuple[str, ...] | None = None

    def _find_active_routes(
        self,
        ensemble_table: EnsembleTable,
    ) -> tuple[str, ...]:
        return tuple(
            stype
            for stype in self.processors
            if any(
                len(group.columns[Stype(stype)]) > 0
                for group in ensemble_table
            )
        )

    def _check_remainder(self, ensemble_table: EnsembleTable) -> None:
        if self.remainder != "error":
            return

        remainder_stypes = [
            stype
            for stype in Stype
            if str(stype) not in self.processors
            and any(len(group.columns[stype]) > 0 for group in ensemble_table)
        ]
        if len(remainder_stypes) == 0:
            return

        names = ", ".join(f"{str(stype)!r}" for stype in remainder_stypes)
        raise ValueError(
            f"Found non-empty input columns for semantic types {names}, but "
            f"{self.__class__.__name__!r} has no route for them. Configure "
            "a processor for each semantic type or set "
            "remainder='passthrough' or remainder='drop'."
        )

    def _fit_ensemble(
        self,
        ensemble_table: EnsembleTable,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        self._check_remainder(ensemble_table)
        self._active_routes = self._find_active_routes(ensemble_table)
        for stype in self._active_routes:
            processor = cast(EnsembleProcessor, self.processors[stype])
            processor.fit_ensemble(
                ensemble_table.select_stypes(stype),
                generator=generator,
            )

    def _fit_transform_ensemble(
        self,
        ensemble_table: EnsembleTable,
        *,
        generator: torch.Generator | None = None,
    ) -> EnsembleTable:
        self._check_remainder(ensemble_table)
        self._active_routes = self._find_active_routes(ensemble_table)
        outputs = [
            cast(
                EnsembleProcessor,
                self.processors[stype],
            ).fit_transform_ensemble(
                ensemble_table.select_stypes(stype),
                generator=generator,
            )
            for stype in self._active_routes
        ]
        if self.remainder == "passthrough":
            outputs.append(self._remainder_ensemble(ensemble_table))
        if len(outputs) == 0:
            return ensemble_table.select_stypes(())
        return EnsembleTable.concatenate_columns(outputs)

    def _transform_ensemble(
        self,
        ensemble_table: EnsembleTable,
    ) -> EnsembleTable:
        self._check_remainder(ensemble_table)

        active_routes = self._active_routes
        if not self.requires_fit or active_routes is None:
            active_routes = self._find_active_routes(ensemble_table)
            self._active_routes = active_routes
        outputs = [
            cast(
                EnsembleProcessor,
                self.processors[stype],
            ).transform_ensemble(ensemble_table.select_stypes(stype))
            for stype in active_routes
        ]
        if self.remainder == "passthrough":
            outputs.append(self._remainder_ensemble(ensemble_table))
        if len(outputs) == 0:
            return ensemble_table.select_stypes(())
        return EnsembleTable.concatenate_columns(outputs)

    def _inverse_transform_ensemble(
        self,
        ensemble_table: EnsembleTable,
    ) -> EnsembleTable:
        if self.remainder == "drop":
            raise ValueError(
                "'StypeDispatch' with remainder='drop' is not invertible"
            )

        self._check_remainder(ensemble_table)
        active_routes = self._active_routes
        if not self.requires_fit or active_routes is None:
            active_routes = self._find_active_routes(ensemble_table)
            self._active_routes = active_routes

        outputs = []
        for stype in active_routes:
            processor = cast(EnsembleProcessor, self.processors[stype])
            inverse = getattr(processor, "inverse_transform_ensemble", None)
            if not callable(inverse):
                raise TypeError(
                    f"Route {stype!r} processor "
                    f"{processor.__class__.__name__!r} is not invertible"
                )
            outputs.append(inverse(ensemble_table.select_stypes(stype)))

        outputs.append(self._remainder_ensemble(ensemble_table))
        return EnsembleTable.concatenate_columns(outputs)

    def _remainder_ensemble(
        self,
        ensemble_table: EnsembleTable,
    ) -> EnsembleTable:
        return ensemble_table.select_stypes(
            tuple(
                stype for stype in Stype if str(stype) not in self.processors
            )
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
