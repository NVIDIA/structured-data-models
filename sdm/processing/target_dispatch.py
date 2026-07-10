from collections.abc import Iterable
from typing import Literal, cast

import torch

from sdm.processing.base import InvertibleMixin, Processor
from sdm.processing.sequential import Sequential
from sdm.stype import Stype
from sdm.tensor import TableTensor


class TargetDispatch(Processor, InvertibleMixin):
    """Route target processing and model-output inversion by target task.

    Fitting selects one route from the raw target semantic type, fits that
    route, and retains its ownership for later transformation. Inverse
    transformation always delegates the complete model output to the selected
    route, independent of the model output's own semantic type.

    Args:
        classification: Invertible processor for one categorical target.
        regression: Invertible processor for one numerical target.
    """

    supported_stypes = frozenset(Stype)

    def __init__(
        self,
        *,
        classification: Processor | Iterable[Processor] | None = None,
        regression: Processor | Iterable[Processor] | None = None,
    ) -> None:
        super().__init__()
        self.processors = torch.nn.ModuleDict()
        for task, processor in (
            ("classification", classification),
            ("regression", regression),
        ):
            if processor is None:
                continue
            if not isinstance(processor, Processor):
                processor = Sequential(*processor)
            if not isinstance(processor, InvertibleMixin):
                raise TypeError(
                    f"'{self.__class__.__name__}' requires an invertible "
                    f"'{task}' route (got "
                    f"'{processor.__class__.__name__}')."
                )
            self.processors[task] = processor

        if len(self.processors) == 0:
            raise ValueError(
                f"'{self.__class__.__name__}' requires at least one route."
            )
        self._task: Literal["classification", "regression"] | None = None

    @property
    def selected(self) -> Processor:
        """The route selected during fitting."""
        if self._task is None:
            raise RuntimeError(
                f"'{self.__class__.__name__}' has no selected target route; "
                "call 'fit()' before."
            )
        return cast(Processor, self.processors[self._task])

    def _fit(self, input: TableTensor) -> None:
        self._task = None
        task = self._resolve_task(input)
        processor = cast(Processor, self.processors[task])
        processor.fit(input)
        self._task = task

    def _transform(self, input: TableTensor) -> TableTensor:
        return self.selected.transform(input)

    def _inverse_transform(self, input: TableTensor) -> TableTensor:
        processor = cast(InvertibleMixin, self.selected)
        return processor.inverse_transform(input)

    def _resolve_task(
        self,
        target: TableTensor,
    ) -> Literal["classification", "regression"]:
        if target.size(-1) != 1:
            raise ValueError(
                "Expected the target to contain exactly one column "
                f"(got {target.size(-1)} columns)."
            )

        if target.categorical.size(-1) == 1:
            task: Literal["classification", "regression"] = "classification"
        elif target.numerical.size(-1) == 1:
            task = "regression"
        else:
            stype = next(
                stype.value
                for stype, columns in target.columns.items()
                if len(columns) > 0
            )
            raise ValueError(
                "Expected the target to be numerical or categorical "
                f"(got '{stype}')."
            )

        if task not in self.processors:
            raise ValueError(
                f"'{self.__class__.__name__}' has no '{task}' route; "
                f"configure {task}=... ."
            )
        return task

    def get_extra_state(self) -> str | None:  # noqa: D102
        return self._task

    def set_extra_state(self, state: str | None) -> None:  # noqa: D102
        if state is not None and state not in self.processors:
            raise ValueError(
                f"Cannot restore unconfigured '{state}' task on "
                f"'{self.__class__.__name__}'."
            )
        self._task = cast(
            Literal["classification", "regression"] | None,
            state,
        )

    def __repr__(self, *, indent: int = 0) -> str:
        reprs = []
        for task, processor in self.processors.items():
            processor = cast(Processor, processor)
            processor_repr = processor.__repr__(indent=indent + 4)
            processor_repr = processor_repr[indent + 4 :]
            reprs.append(f"{' ' * (indent + 2)}{task}: {processor_repr}")
        return (
            f"{' ' * indent}{self.__class__.__name__}(\n"
            + ",\n".join(reprs)
            + f",\n{' ' * indent})"
        )
