from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal, Self

import torch

from sdm import Stype
from sdm.processing import (
    EnsembleInvertibleMixin,
    EnsembleProcessor,
    Identity,
    Processor,
    TableDispatch,
    TaskDispatch,
)
from sdm.tensor import EnsembleTable


class _TaskResolver(EnsembleProcessor, EnsembleInvertibleMixin):
    """Resolve linked task dispatchers while fitting a recipe target."""

    supported_stypes = frozenset(Stype)

    def __init__(
        self,
        processor: EnsembleProcessor,
        task_dispatchers: tuple[TaskDispatch, ...],
    ) -> None:
        super().__init__()
        self.processor = processor
        self._task_dispatchers = task_dispatchers

    def _fit_ensemble(
        self,
        ensemble_table: EnsembleTable,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        self._fit_transform_ensemble(ensemble_table, generator=generator)

    def _fit_transform_ensemble(
        self,
        ensemble_table: EnsembleTable,
        *,
        generator: torch.Generator | None = None,
    ) -> EnsembleTable:

        ensemble_table = self.processor.fit_transform_ensemble(
            ensemble_table, generator=generator
        )

        tasks: set[Literal["classification", "regression"]] = set()
        for group in ensemble_table:
            if group.size(-1) != 1:
                raise ValueError(
                    "Expected the transformed target to contain exactly one "
                    f"column (got {group.size(-1)} columns)"
                )
            if group.numerical.size(-1) == 1:
                tasks.add("regression")
            elif group.categorical.size(-1) == 1:
                tasks.add("classification")
            else:
                stypes = ", ".join(
                    f"{str(stype)!r}" for stype in group.active_stypes
                )
                raise ValueError(
                    "Expected the transformed target to contain exactly one "
                    f"numerical or categorical column (got {stypes})"
                )

        if len(tasks) != 1:
            raise ValueError(
                "'Recipe.target' must resolve to a single task type across "
                "ensemble members"
            )

        task = next(iter(tasks))
        for task_dispatcher in self._task_dispatchers:
            task_dispatcher._task = task

        return ensemble_table

    def _transform_ensemble(
        self,
        ensemble_table: EnsembleTable,
    ) -> EnsembleTable:
        return self.processor.transform_ensemble(ensemble_table)

    def _inverse_transform_ensemble(
        self,
        ensemble_table: EnsembleTable,
    ) -> EnsembleTable:
        if not isinstance(self.processor, EnsembleInvertibleMixin):
            raise AttributeError(
                f"{self.processor.__class__.__name__!r} object has no "
                "attribute 'inverse_transform_ensemble'"
            )
        return self.processor.inverse_transform_ensemble(ensemble_table)

    def __repr__(self, *, indent: int = 0) -> str:
        return self.processor.__repr__(indent=indent)


@dataclass(init=False, repr=False)
class Recipe:
    """Processing contract around an external model boundary.

    A recipe bundles three processing pipelines, one per role the data plays
    relative to the model:

    - ``features``: model inputs, transformed before the model.
    - ``target``: labels transformed forward before the model. Regression
      predictions are inverted through this pipeline; classification outputs
      are reconstructed from the fitted target categories instead.
    - ``output``: transforms member outputs after they have been mapped to a
      common class or target space and stacked as ``[E, ..., R, O]``. An
      explicit dimension-changing step such as
      :class:`~sdm.processing.ReduceEstimators` removes ``E``; without one,
      the output remains stacked. Steps before the reducer must support
      stacked outputs, while steps after it receive already-reduced outputs.

    Each pipeline exposes ``fit``/``transform``/``fit_transform`` and, when its
    steps are invertible, ``inverse_transform``. Call them directly, e.g.
    ``recipe.features.transform(table)`` or
    ``recipe.target.inverse_transform(prediction)``. Recipes do not infer each
    step's non-finite input contract; order steps so values are imputed before
    processors that do not explicitly document non-finite support. When
    ``output`` contains :class:`~sdm.processing.TaskDispatch`, fitting
    ``target`` also selects its task-specific output route.

    Copy a task-aware recipe as a whole so its target remains connected to the
    output dispatchers.

    Bind a recipe to context data with :meth:`bind` to obtain a reusable
    execution for query, inverse-target, and output transforms.

    Args:
        features: Steps applied to model inputs before the model.
        target: Steps applied to labels. Invertible numerical target steps map
            regression output back to the original space.
        output: Steps applied to stacked member outputs after member-local
            mappings. Estimator reduction, when desired, is an explicit step
            in this pipeline.
    """

    features: EnsembleProcessor
    target: EnsembleProcessor
    output: EnsembleProcessor

    def __init__(
        self,
        features: Processor | Iterable[Processor] | None = None,
        target: Processor | Iterable[Processor] | None = None,
        output: Processor | Iterable[Processor] | None = None,
    ) -> None:

        self.features = EnsembleProcessor.as_processor(
            Identity() if features is None else features
        )
        self.target = EnsembleProcessor.as_processor(
            Identity() if target is None else target
        )
        self.output = EnsembleProcessor.as_processor(
            Identity() if output is None else output
        )

        self._validate_target()
        self._validate_output()

        task_dispatchers = tuple(
            m for m in self.features.modules() if isinstance(m, TaskDispatch)
        ) + tuple(
            m for m in self.output.modules() if isinstance(m, TaskDispatch)
        )
        if len(task_dispatchers) > 0:
            self.target = _TaskResolver(self.target, task_dispatchers)

    def prepend_features(self, processor: object) -> Self:
        """Prepend a processor to the feature pipeline."""
        self.features = processor + self.features
        return self

    def append_features(self, processor: object) -> Self:
        """Append a processor to the feature pipeline."""
        self.features = self.features + processor
        return self

    def prepend_target(self, processor: object) -> Self:
        """Prepend a processor to the target pipeline."""
        self.target = processor + self.target
        self._validate_target()
        return self

    def append_target(self, processor: object) -> Self:
        """Append a processor to the target pipeline."""
        self.target = self.target + processor
        self._validate_target()
        return self

    def prepend_output(self, processor: object) -> Self:
        """Prepend a processor to the output pipeline."""
        self.output = processor + self.output
        self._validate_output()
        return self

    def append_output(self, processor: object) -> Self:
        """Append a processor to the output pipeline."""
        self.output = self.output + processor
        self._validate_output()
        return self

    def _validate_target(self) -> None:
        if any(isinstance(m, TaskDispatch) for m in self.target.modules()):
            raise ValueError(
                "'TaskDispatch' is not supported in 'Recipe.target'"
            )

    def _validate_output(self) -> None:
        if self.output.requires_fit:
            raise ValueError("'Recipe.output' should not require fitting")

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(\n"
            f"  features={self.features.__repr__(indent=2)[2:]},\n"
            f"  target={self.target.__repr__(indent=2)[2:]},\n"
            f"  output={self.output.__repr__(indent=2)[2:]},\n"
            ")"
        )
