from collections.abc import Iterable
from dataclasses import dataclass

from sdm.processing.base import InvertibleMixin, Processor
from sdm.processing.sequential import Sequential
from sdm.processing.task_dispatch import TaskDispatch
from sdm.stype import Stype
from sdm.tensor import TableTensor


class _TaskResolver(Processor, InvertibleMixin):
    supported_stypes = frozenset(Stype)

    def __init__(
        self,
        processor: Processor,
        task_dispatchers: tuple[TaskDispatch, ...],
    ) -> None:
        super().__init__()
        self.processor = processor
        # A tuple keeps these modules out of target's PyTorch module tree.
        self._task_dispatchers = task_dispatchers

    def _fit(self, input: TableTensor) -> None:
        self._fit_target(input)

    def _fit_target(self, input: TableTensor) -> TableTensor:
        self._fitted = False
        for task_dispatcher in self._task_dispatchers:
            task_dispatcher._reset()

        succeeded = False
        try:
            target = self.processor.fit_transform(input)
            for task_dispatcher in self._task_dispatchers:
                task_dispatcher._resolve(target)
            succeeded = True
            return target
        finally:
            if not succeeded:
                for task_dispatcher in self._task_dispatchers:
                    task_dispatcher._reset()

    def fit_transform(self, input: TableTensor) -> TableTensor:
        self._check_supported_stypes(input)
        target = self._fit_target(input)
        self._fitted = True
        return target

    def _transform(self, input: TableTensor) -> TableTensor:
        return self.processor.transform(input)

    def _inverse_transform(self, input: TableTensor) -> TableTensor:
        fn = getattr(self.processor, "inverse_transform", None)
        if not callable(fn):
            raise AttributeError(
                f"'{self.processor.__class__.__name__}' object has no "
                "attribute 'inverse_transform'"
            )
        return fn(input)

    def __repr__(self, *, indent: int = 0) -> str:
        processor = self.processor.__repr__(indent=indent + 2)
        return (
            f"{' ' * indent}{self.__class__.__name__}(\n"
            f"{processor},\n"
            f"{' ' * indent})"
        )


@dataclass(frozen=True, init=False, repr=False)
class Recipe:
    """Processing contract around an external model boundary.

    A recipe bundles three :class:`~sdm.processing.Processor` pipelines, one
    per role the data plays relative to the model:

    - ``features``: model inputs, transformed before the model.
    - ``target``: labels, transformed forward before the model and inverted
      after it (predictions back to the original space).
    - ``output``: shape-preserving cleanup of the model output.

    Each pipeline exposes ``fit``/``transform``/``fit_transform`` and, when its
    steps are invertible, ``inverse_transform``. Call them directly, e.g.
    ``recipe.features.transform(table)`` or
    ``recipe.target.inverse_transform(prediction)``. When ``output`` contains
    :class:`~sdm.processing.TaskDispatch`, fitting ``target`` also selects its
    task-specific output route.

    Copy a task-aware recipe as a whole so its target remains connected to the
    output dispatchers.

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

        all_task_dispatchers = tuple(
            module
            for module in output.modules()
            if isinstance(module, TaskDispatch)
        )
        if isinstance(output, TaskDispatch):
            task_dispatchers = (output,)
        elif isinstance(output, Sequential):
            task_dispatchers = tuple(
                step for step in output.steps if isinstance(step, TaskDispatch)
            )
        else:
            task_dispatchers = ()

        if set(all_task_dispatchers) != set(task_dispatchers):
            raise ValueError(
                "'TaskDispatch' must be a direct step in 'Recipe.output'; "
                "nested task dispatch is not supported."
            )

        if len(task_dispatchers) > 0:
            target = _TaskResolver(
                processor=target,
                task_dispatchers=task_dispatchers,
            )

        object.__setattr__(self, "features", features)
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "output", output)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"  features={self.features.__repr__(indent=2)[2:]},\n"
            f"  target={self.target.__repr__(indent=2)[2:]},\n"
            f"  output={self.output.__repr__(indent=2)[2:]},\n"
            ")"
        )
