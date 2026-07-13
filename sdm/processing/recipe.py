from collections.abc import Iterable
from dataclasses import dataclass

from typing_extensions import Self

from sdm.processing.base import InvertibleMixin, Processor
from sdm.processing.sequential import Sequential
from sdm.processing.task_dispatch import TaskDispatch
from sdm.stype import Stype
from sdm.tensor import TableTensor


class _TaskResolver(Processor, InvertibleMixin):
    """Resolve linked output dispatchers while fitting a recipe target.

    The wrapped target processor is a registered child module. Output
    dispatchers stay in a plain tuple so they remain registered only under
    ``Recipe.output``.
    """

    supported_stypes = frozenset(Stype)

    def __init__(
        self,
        processor: Processor,
        task_dispatchers: tuple[TaskDispatch, ...],
    ) -> None:
        super().__init__()
        self.processor = processor
        self._task_dispatchers = task_dispatchers

    def fit(self, input: TableTensor) -> Self:
        self.fit_transform(input)
        return self

    def fit_transform(self, input: TableTensor) -> TableTensor:
        self._check_supported_stypes(input)
        self._fitted = False
        for task_dispatcher in self._task_dispatchers:
            task_dispatcher._reset()

        succeeded = False
        try:
            target = self.processor.fit_transform(input)
            for task_dispatcher in self._task_dispatchers:
                task_dispatcher._resolve(target)
            self._fitted = True
            succeeded = True
            return target
        finally:
            if not succeeded:
                for task_dispatcher in self._task_dispatchers:
                    task_dispatcher._reset()

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
        return self.processor.__repr__(indent=indent)


@dataclass(frozen=True, init=False, repr=False)
class Recipe:
    """Processing contract around an external model boundary.

    A recipe bundles three processing pipelines, one per role the data plays
    relative to the model:

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

        # TODO: Support TaskDispatch in features after defining task-aware
        # feature fit ordering.
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

        # Common output steps can remain adjacent; nesting would require
        # defining whether dispatchers in inactive branches are resolved.
        task_dispatcher_entries = tuple(
            (path, module)
            for path, module in output.named_modules(remove_duplicate=False)
            if isinstance(module, TaskDispatch)
        )
        if isinstance(output, TaskDispatch):
            direct_paths = {""}
        elif isinstance(output, Sequential):
            direct_paths = {
                str(index)
                for index, step in enumerate(output.steps)
                if isinstance(step, TaskDispatch)
            }
        else:
            direct_paths = set()

        nested_paths = tuple(
            path
            for path, _ in task_dispatcher_entries
            if path not in direct_paths
        )
        if len(nested_paths) > 0:
            locations = ", ".join(repr(path) for path in nested_paths)
            raise ValueError(
                "'TaskDispatch' must be a direct step in 'Recipe.output'; "
                f"nested task dispatch was found at {locations}."
            )

        task_dispatchers = tuple(
            module
            for path, module in task_dispatcher_entries
            if path in direct_paths
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
            f"{self.__class__.__name__}(\n"
            f"  features={self.features.__repr__(indent=2)[2:]},\n"
            f"  target={self.target.__repr__(indent=2)[2:]},\n"
            f"  output={self.output.__repr__(indent=2)[2:]},\n"
            ")"
        )
