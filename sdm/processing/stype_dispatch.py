from collections.abc import Iterable
from typing import Literal, cast

import torch
from torch import Tensor

from sdm import Stype
from sdm.processing.base import Processor
from sdm.processing.sequential import Sequential
from sdm.tensor import TableTensor


class StypeDispatch(Processor):
    r"""Apply separate processor pipelines to columns grouped by semantic type.

    For each configured route, the matching columns are selected into a
    :class:`TableTensor` and passed to that processor. A route may change
    column values, names, count, or order. Route outputs are concatenated in
    semantic type order. With the default passthrough behavior, unconfigured
    semantic types follow in input order.

    Args:
        numerical: Processor route for numerical columns. An iterable is
            normalized to :class:`~sdm.processing.Sequential`.
        categorical: Processor route for categorical columns. An iterable is
            normalized to :class:`~sdm.processing.Sequential`.
        datetime: Processor route for datetime columns. An iterable is
            normalized to :class:`~sdm.processing.Sequential`.
        id: Processor route for identifier columns. An iterable is normalized
            to :class:`~sdm.processing.Sequential`.
        remainder: How to handle non-empty semantic types without a configured
            route. ``"passthrough"`` keeps them unchanged and is the default,
            ``"drop"`` removes them, and ``"error"`` raises.
    """

    supported_stypes = frozenset(Stype)

    def __init__(
        self,
        *,
        numerical: Processor | Iterable[Processor] | None = None,
        categorical: Processor | Iterable[Processor] | None = None,
        datetime: Processor | Iterable[Processor] | None = None,
        id: Processor | Iterable[Processor] | None = None,
        remainder: Literal["passthrough", "drop", "error"] = "passthrough",
    ) -> None:
        super().__init__()
        self.processors = torch.nn.ModuleDict()
        for stype, processor in (
            (Stype.numerical, numerical),
            (Stype.categorical, categorical),
            (Stype.datetime, datetime),
            (Stype.id, id),
        ):
            if processor is None:
                continue
            if not isinstance(processor, Processor):
                processor = Sequential(*processor)
            self.processors[stype.value] = processor

        self.remainder = remainder
        self.requires_fit = any(
            processor.requires_fit for processor in self.processors.values()
        )

    def _check_remainder(self, remainder_stypes: list[Stype]) -> None:
        if self.remainder != "error" or len(remainder_stypes) == 0:
            return

        names = ", ".join(f"'{stype.value}'" for stype in remainder_stypes)
        raise ValueError(
            f"Found non-empty input columns for semantic types {names}, but "
            f"'{self.__class__.__name__}' has no route for them. Configure "
            "a processor for each semantic type or set "
            "remainder='passthrough' or remainder='drop'."
        )

    def _fit(self, input: TableTensor) -> None:
        remainder_stypes = [
            stype
            for stype, columns in input.columns.items()
            if stype.value not in self.processors and len(columns) > 0
        ]
        self._check_remainder(remainder_stypes)
        for stype, processor in self.processors.items():
            processor = cast(Processor, processor)
            route_input = input.select_stypes(stype)
            if route_input.size(-1) == 0:
                continue
            processor.fit(route_input)

    def _transform(self, input: TableTensor) -> TableTensor:
        remainder_stypes = [
            stype
            for stype, columns in input.columns.items()
            if stype.value not in self.processors and len(columns) > 0
        ]
        self._check_remainder(remainder_stypes)

        outputs: list[TableTensor] = []
        for stype, processor in self.processors.items():
            processor = cast(Processor, processor)
            route_input = input.select_stypes(stype)
            if route_input.size(-1) == 0:
                continue
            outputs.append(processor.transform(route_input))
        if self.remainder == "passthrough":
            outputs.extend(
                input.select_stypes(stype) for stype in remainder_stypes
            )
        if len(outputs) == 0:
            return input.select_columns(())
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
            processor_repr = processor.__repr__(indent=indent + 4)
            reprs.append(f"{' ' * (indent + 2)}{stype}: {processor_repr}")
        return (
            f"{' ' * indent}{self.__class__.__name__}(\n"
            + ",\n".join(reprs)
            + f",\n{' ' * indent})"
        )
