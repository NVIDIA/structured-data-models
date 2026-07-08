from collections.abc import Iterable, Mapping
from typing import Literal, cast

import torch
from torch import Tensor

from sdm import Stype, StypeLike
from sdm.processing.base import Processor
from sdm.processing.sequential import Sequential
from sdm.tensor import TableTensor


class StypeDispatch(Processor):
    r"""Apply separate processor pipelines to columns grouped by semantic type.

    For each ``processors`` entry, the matching columns are selected into a
    :class:`TableTensor` and passed to that processor. A route may change
    column values, names, count, or order. Route outputs are concatenated in
    mapping insertion order. With the default passthrough behavior,
    unconfigured semantic types follow in input order.

    Args:
        processors: Mapping from semantic type to a single processor or an
            iterable of processors. Iterable routes are normalized to
            :class:`~sdm.processing.Sequential`.
        remainder: How to handle non-empty semantic types without a configured
            route. ``"passthrough"`` keeps them unchanged and is the default,
            ``"drop"`` removes them, and ``"error"`` raises.
    """

    supported_stypes = frozenset(Stype)

    def __init__(
        self,
        processors: Mapping[
            StypeLike,
            Processor | Iterable[Processor],
        ],
        *,
        remainder: Literal["passthrough", "drop", "error"] = "passthrough",
    ) -> None:
        super().__init__()
        if remainder not in ("passthrough", "drop", "error"):
            raise ValueError(
                "Expected 'remainder' to be one of "
                "'passthrough', 'drop', or 'error' "
                f"(got {remainder!r})"
            )

        self.processors = torch.nn.ModuleDict()
        for stype, processor in processors.items():
            if not isinstance(processor, Processor):
                processor = Sequential(*processor)
            self.processors[Stype(stype).value] = processor

        self.remainder = remainder
        self.requires_fit = any(
            processor.requires_fit for processor in self.processors.values()
        )

    def _processors_by_stype(self) -> Iterable[tuple[Stype, Processor]]:
        for stype, processor in self.processors.items():
            yield Stype(stype), cast(Processor, processor)

    def _remainder_stypes(self, input: TableTensor) -> list[Stype]:
        configured = {stype for stype, _ in self._processors_by_stype()}
        return [
            stype
            for stype, columns in input.columns.items()
            if stype not in configured and len(columns) > 0
        ]

    def _check_remainder(self, remainder_stypes: list[Stype]) -> None:
        if self.remainder != "error" or len(remainder_stypes) == 0:
            return

        names = ", ".join(f"'{stype.value}'" for stype in remainder_stypes)
        raise ValueError(
            f"'{self.__class__.__name__}' has no route for {names} "
            "columns; configure a processor or set "
            "remainder='passthrough' or remainder='drop'."
        )

    def _remainder_outputs(self, input: TableTensor) -> list[TableTensor]:
        remainder_stypes = self._remainder_stypes(input)
        self._check_remainder(remainder_stypes)
        if self.remainder == "passthrough":
            return [input.select_stypes(stype) for stype in remainder_stypes]
        return []

    def _merge_outputs(
        self,
        input: TableTensor,
        outputs: list[TableTensor],
    ) -> TableTensor:
        if len(outputs) == 0:
            return input.select_columns(())
        return cast(
            TableTensor,
            torch.cat(cast(list[Tensor], outputs), dim=-1),
        )

    def _fit(self, input: TableTensor) -> None:
        # TODO: Move this input-aware check to pipeline validation.
        self._check_remainder(self._remainder_stypes(input))
        for stype, processor in self._processors_by_stype():
            route_input = input.select_stypes(stype)
            if route_input.size(-1) == 0:
                continue
            processor.fit(route_input)

    def _transform(self, input: TableTensor) -> TableTensor:
        remainder_outputs = self._remainder_outputs(input)
        outputs: list[TableTensor] = []
        for stype, processor in self._processors_by_stype():
            route_input = input.select_stypes(stype)
            if route_input.size(-1) == 0:
                continue
            outputs.append(processor.transform(route_input))
        outputs.extend(remainder_outputs)
        return self._merge_outputs(input, outputs)

    def fit_transform(self, input: TableTensor) -> TableTensor:  # noqa: D102
        remainder_outputs = self._remainder_outputs(input)
        outputs: list[TableTensor] = []
        for stype, processor in self._processors_by_stype():
            route_input = input.select_stypes(stype)
            if route_input.size(-1) == 0:
                continue
            outputs.append(processor.fit_transform(route_input))
        outputs.extend(remainder_outputs)
        self._fitted = True
        return self._merge_outputs(input, outputs)

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
