from typing import Literal, cast

import torch

from sdm.processing.base import Processor
from sdm.processing.ensemble import (
    EnsembleFitContext,
    EnsembleProcessor,
    as_ensemble_processor,
)
from sdm.stype import Stype
from sdm.tensor import EnsembleTable, TableTensor


class TaskDispatch(EnsembleProcessor):
    """Route model output by the transformed target's semantic type.

    When used in :attr:`Recipe.output <sdm.processing.Recipe.output>`, fitting
    :attr:`Recipe.target <sdm.processing.Recipe.target>` resolves the route
    from the final transformed target. One numerical column selects regression
    and one categorical column selects classification. Routes must be stateless
    because output processing has no fitting data of its own. Nested
    dispatchers are resolved only within the selected route.

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
        classification: object = None,
        regression: object = None,
    ) -> None:
        super().__init__()
        self.processors = torch.nn.ModuleDict()
        for task, processor in (
            ("classification", classification),
            ("regression", regression),
        ):
            if processor is None:
                continue
            processor = Processor.as_processor(processor)
            if processor.requires_fit:
                raise ValueError(
                    f"{self.__class__.__name__!r} requires stateless routes, "
                    f"but the {task!r} route requires fit."
                )
            self.processors[task] = processor

        if len(self.processors) == 0:
            raise ValueError(
                f"{self.__class__.__name__!r} requires at least one route."
            )

        self._task: Literal["classification", "regression"] | None = None

    @classmethod
    def _roots(
        cls,
        module: torch.nn.Module,
    ) -> tuple["TaskDispatch", ...]:
        if isinstance(module, cls):
            return (module,)
        return tuple(
            dispatcher
            for child in module.children()
            for dispatcher in cls._roots(child)
        )

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
                f"categorical (got {stype!r})."
            )

        if task not in self.processors:
            raise ValueError(
                f"{self.__class__.__name__!r} has no {task!r} route; "
                f"configure {task}=... or use 'Identity()' for a no-op."
            )
        self._task = task
        selected = cast(Processor, self.processors[task])
        for dispatcher in self._roots(selected):
            dispatcher._resolve(target)

    def _reset(self) -> None:
        self._task = None
        for module in self.processors.modules():
            if isinstance(module, TaskDispatch):
                module._task = None

    def _transform(self, table: TableTensor) -> TableTensor:
        if self._task is None:
            raise RuntimeError(
                f"{self.__class__.__name__!r} has no resolved task; call "
                "'recipe.target.fit()' before transforming model output."
            )
        processor = cast(Processor, self.processors[self._task])
        return processor.transform(table)

    def _ensemble_route(self) -> EnsembleProcessor:
        if self._task is None:
            raise RuntimeError("TaskDispatch has no resolved task.")
        route = as_ensemble_processor(
            cast(Processor, self.processors[self._task])
        )
        self.processors[self._task] = route
        return route

    def fit_transform_ensemble(
        self,
        table: EnsembleTable,
        *,
        context: EnsembleFitContext,
    ) -> EnsembleTable:
        r"""Fit and transform the resolved task route."""
        return self._ensemble_route().fit_transform_ensemble(
            table,
            context=context.child(self._task or "task"),
        )

    def transform_ensemble(self, table: EnsembleTable) -> EnsembleTable:
        r"""Transform through the resolved task route."""
        return self._ensemble_route().transform_ensemble(table)

    def get_extra_state(self) -> str | None:
        r""":meta private:"""  # noqa: D415
        return self._task

    def set_extra_state(self, state: str | None) -> None:
        r""":meta private:"""  # noqa: D415
        if state is not None and state not in self.processors:
            raise ValueError(
                f"Cannot restore unconfigured {state!r} task on "
                f"{self.__class__.__name__!r}."
            )
        self._task = cast(
            Literal["classification", "regression"] | None,
            state,
        )

    def __repr__(self, *, indent: int = 0) -> str:
        reprs = []
        for task, processor in self.processors.items():
            processor = cast(Processor, processor)
            processor_repr = processor.__repr__(indent=indent + 2)
            processor_repr = processor_repr[indent + 2 :]
            reprs.append(f"{' ' * (indent + 2)}{task}: {processor_repr}")
        return (
            f"{' ' * indent}{self.__class__.__name__}(\n"
            + ",\n".join(reprs)
            + f",\n{' ' * indent})"
        )
