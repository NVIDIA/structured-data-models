from collections.abc import Iterable
from typing import Literal, cast

import torch

from sdm.processing.base import Processor
from sdm.processing.sequential import Sequential
from sdm.stype import Stype
from sdm.tensor import TableTensor


class TaskDispatch(Processor):
    """Apply an output processor selected from the transformed target type.

    The enclosing :class:`~sdm.processing.Recipe` resolves the route when its
    target pipeline returns exactly one numerical or categorical column.
    Numerical targets select regression and categorical targets select
    classification. Routes must be stateless because output processing has no
    fitting data of its own.

    Args:
        classification: Output processor for categorical targets. An iterable
            is normalized to :class:`~sdm.processing.Sequential`.
        regression: Output processor for numerical targets. An iterable is
            normalized to :class:`~sdm.processing.Sequential`.
    """

    supported_stypes = frozenset(Stype)
    requires_fit = False

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
            if processor.requires_fit:
                raise ValueError(
                    f"'{self.__class__.__name__}' requires stateless routes, "
                    f"but the '{task}' route requires fit."
                )
            self.processors[task] = processor

        if len(self.processors) == 0:
            raise ValueError(
                f"'{self.__class__.__name__}' requires at least one route."
            )

        self._task: Literal["classification", "regression"] | None = None

    def _resolve(self, target: TableTensor) -> None:
        self._reset()
        if target.size(-1) != 1:
            raise ValueError(
                "Expected the transformed target to contain exactly one "
                f"column (got {target.size(-1)} columns)."
            )

        if target.numerical.size(-1) == 1:
            task = "regression"
        elif target.categorical.size(-1) == 1:
            task = "classification"
        else:
            stype = next(
                stype.value
                for stype, columns in target.columns.items()
                if len(columns) > 0
            )
            raise ValueError(
                "Expected the transformed target to be numerical or "
                f"categorical (got '{stype}')."
            )

        if task not in self.processors:
            raise ValueError(
                f"'{self.__class__.__name__}' has no '{task}' route; "
                f"configure {task}=... or use 'Identity()' for a no-op."
            )
        self._task = task

    def _reset(self) -> None:
        self._task = None

    def _transform(self, input: TableTensor) -> TableTensor:
        if self._task is None:
            raise RuntimeError(
                f"'{self.__class__.__name__}' has no resolved task; call "
                "'Recipe.fit_transform()' before transforming model output."
            )
        processor = cast(Processor, self.processors[self._task])
        return processor.transform(input)

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
            reprs.append(f"{' ' * (indent + 2)}{task}: {processor_repr}")
        task = f"'{self._task}'" if self._task is not None else "None"
        reprs.append(f"{' ' * (indent + 2)}task: {task}")
        return (
            f"{' ' * indent}{self.__class__.__name__}(\n"
            + ",\n".join(reprs)
            + f",\n{' ' * indent})"
        )
