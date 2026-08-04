from collections.abc import Mapping, Sequence
from typing import Literal, cast

import torch

from sdm.processing.base import Processor
from sdm.processing.ensemble import (
    EnsembleInvertibleMixin,
    EnsembleProcessor,
    EnsembleProcessorAdapter,
)
from sdm.stype import Stype
from sdm.tensor import EnsembleTable, TableTensor


class Choice(EnsembleProcessor, EnsembleInvertibleMixin):
    """Delegate each table or ensemble member to one selected option.

    The option is drawn when the processor is fitted; pass ``generator``
    to ``fit()`` to make it reproducible. The generator is also passed on
    to fit the drawn option. Only the drawn option is fitted; refitting
    draws again. Ensemble execution selects one option per member.

    Args:
        args: Sequence of candidate processors or stateless callables. Each
            callable accepts and returns a :class:`~sdm.tensor.TableTensor`.
        selection: Selection strategy. Round-robin assigns options by ensemble
            member position and selects the first option for a single table.
    """

    supported_stypes = frozenset(Stype)

    def __init__(
        self,
        *args: object,
        selection: Literal["random", "round_robin"] = "random",
    ) -> None:
        super().__init__()
        if len(args) == 0:
            raise ValueError("'Choice' requires at least one option.")
        if selection not in {"random", "round_robin"}:
            raise ValueError("Expected 'random' or 'round_robin' selection.")
        self.options = torch.nn.ModuleList(
            Processor.as_processor(arg) for arg in args
        )
        self.selection = selection
        self._selections: tuple[int, ...] = ()

    @property
    def selected(self) -> Processor:
        """The drawn option."""
        if len(self._selections) == 0:
            raise RuntimeError(
                f"{self.__class__.__name__!r} has no drawn option; "
                "call 'fit()' before."
            )
        if len(self._selections) > 1:
            raise RuntimeError(
                f"{self.__class__.__name__!r} has multiple drawn options."
            )

        option = cast(Processor, self.options[self._selections[0]])
        if not isinstance(option, EnsembleProcessorAdapter):
            return option
        if len(option.processors) == 0:
            return option.template
        return cast(Processor, option.processors[0])

    @staticmethod
    def _merge_ensemble_outputs(
        selections: Sequence[int],
        outputs: Mapping[int, EnsembleTable],
    ) -> EnsembleTable:
        tables: list[TableTensor] = []
        locations: dict[tuple[int, tuple[int, int]], int] = {}
        member_table_ids = []
        for member_id, option in enumerate(selections):
            output = outputs[option]
            location = output._member_location(member_id)
            key = (option, location)
            if key not in locations:
                locations[key] = len(tables)
                tables.append(output.table(member_id))
            member_table_ids.append(locations[key])

        return EnsembleTable.from_tables(
            tables=tables,
            member_table_ids=member_table_ids,
        )

    def _select_ensemble_options(
        self,
        ensemble_table: EnsembleTable,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        for index, option in enumerate(tuple(self.options)):
            self.options[index] = EnsembleProcessorAdapter.adapt(
                cast(Processor, option)
            )

        if self.selection == "round_robin":
            self._selections = tuple(
                member_id % len(self.options)
                for member_id in range(ensemble_table.num_members)
            )
        else:
            device = (
                next(iter(ensemble_table)).device
                if generator is None
                else generator.device
            )
            self._selections = tuple(
                torch.randint(
                    len(self.options),
                    (ensemble_table.num_members,),
                    generator=generator,
                    device=device,
                ).tolist()
            )

    def _fit_ensemble(
        self,
        ensemble_table: EnsembleTable,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        self._select_ensemble_options(ensemble_table, generator=generator)
        for option in sorted(set(self._selections)):
            processor = cast(EnsembleProcessor, self.options[option])
            processor.fit_ensemble(ensemble_table, generator=generator)

    def _fit_transform_ensemble(
        self,
        ensemble_table: EnsembleTable,
        *,
        generator: torch.Generator | None = None,
    ) -> EnsembleTable:
        self._select_ensemble_options(ensemble_table, generator=generator)

        # Each option sees stable member positions, including nested choices.
        outputs = {
            option: cast(
                EnsembleProcessor,
                self.options[option],
            ).fit_transform_ensemble(
                ensemble_table,
                generator=generator,
            )
            for option in sorted(set(self._selections))
        }
        return self._merge_ensemble_outputs(self._selections, outputs)

    def _transform_ensemble(
        self,
        ensemble_table: EnsembleTable,
    ) -> EnsembleTable:
        if len(self._selections) != ensemble_table.num_members:
            raise RuntimeError(
                "Choice must be fitted with the same number of ensemble "
                "members before transform."
            )
        outputs = {
            option: cast(
                EnsembleProcessor,
                self.options[option],
            ).transform_ensemble(ensemble_table)
            for option in sorted(set(self._selections))
        }
        return self._merge_ensemble_outputs(self._selections, outputs)

    def _inverse_transform_ensemble(
        self,
        ensemble_table: EnsembleTable,
    ) -> EnsembleTable:
        outputs = {}
        for option in sorted(set(self._selections)):
            processor = cast(EnsembleProcessor, self.options[option])
            inverse = getattr(processor, "inverse_transform_ensemble", None)
            if not callable(inverse):
                raise AttributeError(
                    f"{processor.__class__.__name__!r} object has no "
                    "attribute 'inverse_transform_ensemble'"
                )
            outputs[option] = inverse(ensemble_table)
        return self._merge_ensemble_outputs(self._selections, outputs)

    def __repr__(self, *, indent: int = 0) -> str:
        inner = ",\n".join(
            cast(Processor, option).__repr__(indent=indent + 2)
            for option in self.options
        )
        if self.selection != "random":
            inner += f",\n{' ' * (indent + 2)}selection={self.selection!r}"
        return (
            f"{' ' * indent}{self.__class__.__name__}(\n"
            f"{inner},\n"
            f"{' ' * indent})"
        )
