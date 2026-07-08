from collections.abc import Iterable
from dataclasses import dataclass

from sdm.processing.base import Processor
from sdm.processing.sequential import Sequential
from sdm.processing.task_dispatch import TaskDispatch
from sdm.tensor import TableTensor


@dataclass(frozen=True, init=False, repr=False)
class Recipe:
    """Processing contract around an external model boundary.

    A recipe bundles three :class:`~sdm.processing.Sequential` objects, one per
    role the data plays relative to the model:

    - ``features``: model inputs, transformed before the model.
    - ``target``: labels, transformed forward before the model and inverted
      after it (predictions back to the original space).
    - ``output``: shape-preserving cleanup of the model output.

    Use :meth:`fit_transform` to fit and transform labeled feature and target
    tables together. It transforms the target first so its semantic type can
    resolve any :class:`~sdm.processing.TaskDispatch` in ``output``. Each role
    also exposes its processor methods directly, e.g.
    ``recipe.features.transform(table)`` or
    ``recipe.target.inverse_transform(prediction)``.

    Args:
        features: Steps applied to model inputs before the model.
        target: Steps applied to labels; transformed forward before the model
            and inverted after it.
        output: Steps applied to model output after the target inverse.
    """

    features: Processor
    target: Processor
    output: Processor

    def __init__(
        self,
        features: Processor | Iterable[Processor] | None = None,
        target: Processor | Iterable[Processor] | None = None,
        output: Processor | Iterable[Processor] | None = None,
    ) -> None:

        if features is None:
            features = Sequential()
        elif not isinstance(features, Processor):
            features = Sequential(*features)

        if target is None:
            target = Sequential()
        elif not isinstance(target, Processor):
            target = Sequential(*target)

        if output is None:
            output = Sequential()
        elif not isinstance(output, Processor):
            output = Sequential(*output)

        for role, processor in (
            ("features", features),
            ("target", target),
        ):
            if any(
                isinstance(module, TaskDispatch)
                for module in processor.modules()
            ):
                raise ValueError(
                    f"'TaskDispatch' is only supported in 'Recipe.output' "
                    f"(found in '{role}')."
                )

        object.__setattr__(self, "features", features)
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "output", output)

    def fit_transform(
        self,
        features: TableTensor,
        target: TableTensor,
    ) -> tuple[TableTensor, TableTensor]:
        """Fit and transform labeled features and their target.

        The target is transformed first and resolves task-dependent output
        processors before the feature pipeline is fitted.

        Args:
            features: Feature table with shape ``[R, C]``, where ``R`` is
                the number of labeled rows and ``C`` is the number of columns.
            target: Single-column target table with shape ``[R, 1]``.

        Returns:
            Transformed feature and target tables.
        """
        dispatchers = [
            module
            for module in self.output.modules()
            if isinstance(module, TaskDispatch)
        ]
        for dispatcher in dispatchers:
            dispatcher._reset()

        succeeded = False
        try:
            target = self.target.fit_transform(target)
            for dispatcher in dispatchers:
                dispatcher._resolve(target)
            features = self.features.fit_transform(features)
            succeeded = True
        finally:
            if not succeeded:
                for dispatcher in dispatchers:
                    dispatcher._reset()
        return features, target

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"  features={self.features.__repr__(indent=2)[2:]},\n"
            f"  target={self.target.__repr__(indent=2)[2:]},\n"
            f"  output={self.output.__repr__(indent=2)[2:]},\n"
            ")"
        )
