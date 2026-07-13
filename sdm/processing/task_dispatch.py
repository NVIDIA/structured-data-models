from collections.abc import Iterable
from typing import Literal, cast

import torch
from typing_extensions import Self

from sdm.processing.base import InvertibleMixin, Processor
from sdm.processing.sequential import Sequential
from sdm.stype import Stype
from sdm.tensor import TableTensor


class TaskDispatch(Processor, InvertibleMixin):
    """Route processing by the target task.

    One numerical target column selects the regression route and one
    categorical target column selects the classification route.

    In :attr:`Recipe.target <sdm.processing.Recipe.target>`, fitting resolves
    the route from the fitted target itself and fits the selected route. The
    route transforms the target, and its inverse receives the complete
    numerical model-output head (e.g., class logits or regression quantiles).

    In :attr:`Recipe.output <sdm.processing.Recipe.output>`, fitting
    :attr:`Recipe.target <sdm.processing.Recipe.target>` resolves the route
    from the final transformed target. Output routes must be stateless
    because output processing has no fitting data of its own. Configure
    ``TaskDispatch`` as a direct step in ``Recipe.output``.

    Args:
        classification: Route for categorical targets. An iterable is
            normalized to :class:`~sdm.processing.Sequential`.
        regression: Route for numerical targets. An iterable is normalized
            to :class:`~sdm.processing.Sequential`.
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
            self.processors[task] = processor

        if len(self.processors) == 0:
            raise ValueError(
                f"'{self.__class__.__name__}' requires at least one route."
            )

        self.requires_fit = any(
            cast(Processor, processor).requires_fit
            for processor in self.processors.values()
        )
        self._task: Literal["classification", "regression"] | None = None

    def fit(self, input: TableTensor) -> Self:  # noqa: D102
        # Resolving the route is fitted state, so fitting always resolves,
        # even when every route is stateless.
        self._check_supported_stypes(input)
        self._fit(input)
        self._fitted = True
        return self

    def _fit(self, input: TableTensor) -> None:
        self._resolve(input)
        task = self._task
        assert task is not None
        cast(Processor, self.processors[task]).fit(input)

    def _resolve(self, target: TableTensor) -> None:
        self._reset()
        if target.size(-1) != 1:
            raise ValueError(
                "Expected the target to contain exactly one "
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
                "Expected the target to be numerical or "
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
                "'recipe.target.fit()' before transforming model output."
            )
        processor = cast(Processor, self.processors[self._task])
        return processor.transform(input)

    def _inverse_transform(self, input: TableTensor) -> TableTensor:
        if self._task is None:
            raise RuntimeError(
                f"'{self.__class__.__name__}' has no resolved task; call "
                "'recipe.target.fit()' before inverting model output."
            )
        processor = cast(Processor, self.processors[self._task])
        if not isinstance(processor, InvertibleMixin):
            raise TypeError(
                f"Route '{self._task}' uses non-invertible processor "
                f"'{processor.__class__.__name__}'"
            )
        return processor.inverse_transform(input)

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
