from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field

from torch import Tensor
from typing_extensions import Self

from sdm import Stype
from sdm.processing.base import InvertibleMixin, Processor
from sdm.tensor import TableTensor


class Pipeline:
    """Ordered sequence of processing steps for one recipe phase.

    Args:
        steps: Ordered processing steps. ``None`` creates an empty identity
            pipeline.
    """

    def __init__(
        self,
        steps: Iterable[Processor] | None = None,
    ) -> None:
        self.steps = tuple(_validate_step(step) for step in steps or ())

    def fit(self, table: TableTensor) -> Self:
        """Fit steps in order using the numerical block of ``table``."""
        self._fit(table, phase="pipeline")
        return self

    def _fit(
        self,
        table: TableTensor,
        *,
        phase: str,
    ) -> Self:
        numerical = table.numerical
        last = len(self.steps) - 1
        for position, step in enumerate(self.steps):
            try:
                step.fit(numerical)
                if position != last:
                    numerical = _validate_output(step.transform(numerical))
            except Exception as exc:
                raise _step_error(exc, phase, position, step) from exc
        return self

    def transform(self, table: TableTensor) -> TableTensor:
        """Transform ``table`` by applying steps to its numerical block."""
        return self._transform(table, phase="pipeline")

    def _transform(self, table: TableTensor, *, phase: str) -> TableTensor:
        if len(self.steps) == 0:
            return table
        numerical = table.numerical
        for position, step in enumerate(self.steps):
            try:
                numerical = _validate_output(step.transform(numerical))
            except Exception as exc:
                raise _step_error(exc, phase, position, step) from exc
        return _with_step_numerical(table, numerical, phase, position, step)

    def fit_transform(self, table: TableTensor) -> TableTensor:
        """Fit and transform ``table`` by threading steps in order."""
        return self._fit_transform(table, phase="pipeline")

    def _fit_transform(self, table: TableTensor, *, phase: str) -> TableTensor:
        if len(self.steps) == 0:
            return table
        numerical = table.numerical
        for position, step in enumerate(self.steps):
            try:
                numerical = _validate_output(step.fit_transform(numerical))
            except Exception as exc:
                raise _step_error(exc, phase, position, step) from exc
        return _with_step_numerical(table, numerical, phase, position, step)

    def inverse_transform(self, table: TableTensor) -> TableTensor:
        """Apply invertible steps in reverse order to ``table``."""
        return self._inverse_transform(table, phase="pipeline")

    def _inverse_transform(
        self,
        table: TableTensor,
        *,
        phase: str,
    ) -> TableTensor:
        if len(self.steps) == 0:
            return table
        numerical = table.numerical
        for position, step in reversed(tuple(enumerate(self.steps))):
            if not isinstance(step, InvertibleMixin):
                raise _step_error(
                    TypeError(
                        "Expected invertible step for inverse_transform"
                    ),
                    phase,
                    position,
                    step,
                )
            try:
                numerical = _validate_output(step.inverse_transform(numerical))
            except Exception as exc:
                raise _step_error(exc, phase, position, step) from exc
        return _with_step_numerical(table, numerical, phase, position, step)

    def describe(self) -> str:
        """Return a human-readable step-order summary."""
        if len(self.steps) == 0:
            return "identity"
        return " -> ".join(step.__class__.__name__ for step in self.steps)

    def __len__(self) -> int:
        return len(self.steps)

    def __iter__(self) -> Iterator[Processor]:
        return iter(self.steps)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.describe()})"


@dataclass(frozen=True)
class Recipe:
    """Processing contract around an external model boundary.

    A recipe owns deterministic, two-sided transforms keyed by the data each
    phase operates on:

    - ``features``: model inputs, transformed before the model.
    - ``target``: labels, transformed forward before the model and inverted
      after it (predictions back to the original space).
    - ``output``: shape-preserving cleanup of the model output.

    Args:
        features: Steps applied to model inputs before the model.
        target: Steps applied to labels; transformed forward before the model
            and inverted after it.
        output: Steps applied to model output after the target inverse.
    """

    features: Pipeline = field(default_factory=Pipeline)
    target: Pipeline = field(default_factory=Pipeline)
    output: Pipeline = field(default_factory=Pipeline)

    def __init__(
        self,
        features: Iterable[Processor] | None = None,
        target: Iterable[Processor] | None = None,
        output: Iterable[Processor] | None = None,
    ) -> None:
        object.__setattr__(self, "features", _coerce_pipeline(features))
        object.__setattr__(self, "target", _coerce_pipeline(target))
        object.__setattr__(self, "output", _coerce_pipeline(output))

    def fit_features(self, table: TableTensor) -> Self:
        """Fit the feature phase on ``table`` and return this recipe."""
        self.features._fit(table, phase="features")
        return self

    def transform_features(self, table: TableTensor) -> TableTensor:
        """Transform model inputs before an external model call."""
        return self.features._transform(table, phase="features")

    def fit_transform_features(self, table: TableTensor) -> TableTensor:
        """Fit and transform model inputs before an external model call."""
        return self.features._fit_transform(table, phase="features")

    def fit_target(self, table: TableTensor) -> Self:
        """Fit the target phase on ``table`` and return this recipe."""
        self.target._fit(table, phase="target")
        return self

    def transform_target(self, table: TableTensor) -> TableTensor:
        """Transform labels into model space before an external model call."""
        return self.target._transform(table, phase="target")

    def fit_transform_target(self, table: TableTensor) -> TableTensor:
        """Fit and transform labels into model space before the model."""
        return self.target._fit_transform(table, phase="target")

    def inverse_transform_target(self, table: TableTensor) -> TableTensor:
        """Run target inverse conversion after an external model call."""
        return self.target._inverse_transform(table, phase="target")

    def transform_output(self, table: TableTensor) -> TableTensor:
        """Run shape-preserving cleanup after an external model call."""
        return self.output._transform(table, phase="output")

    def fit_transform(
        self,
        features: TableTensor,
        target: TableTensor,
    ) -> tuple[TableTensor, TableTensor]:
        """Fit and transform training ``features`` and ``target`` at once.

        Convenience for the common training step. Returns the transformed
        ``(features, target)`` tables, ready to feed the model and its loss.
        """
        return (
            self.fit_transform_features(features),
            self.fit_transform_target(target),
        )

    def describe(self) -> str:
        """Return a human-readable summary without running inference."""
        return (
            "Recipe(\n"
            f"  features: {self.features.describe()}\n"
            f"  target: {self.target.describe()}\n"
            f"  output: {self.output.describe()}\n"
            ")"
        )

    def __repr__(self) -> str:
        return self.describe()


def _coerce_pipeline(
    value: Iterable[Processor] | None,
) -> Pipeline:
    if isinstance(value, Pipeline):
        return value
    return Pipeline(value)


def _validate_step(step: object) -> Processor:
    if not isinstance(step, Processor):
        raise TypeError(
            f"Expected a Processor step (got '{step.__class__.__name__}')"
        )
    return step


def _validate_output(output: object) -> Tensor:
    if not isinstance(output, Tensor):
        raise TypeError(
            "Expected the step to return a Tensor for the numerical block "
            f"(got '{type(output).__name__}')"
        )
    return output


def _with_step_numerical(
    table: TableTensor,
    numerical: Tensor,
    phase: str,
    position: int,
    step: Processor,
) -> TableTensor:
    try:
        return _with_numerical(table, numerical)
    except Exception as exc:
        raise _step_error(exc, phase, position, step) from exc


def _with_numerical(table: TableTensor, numerical: Tensor) -> TableTensor:
    if numerical is table.numerical:
        return table
    return table.__class__(
        columns={
            Stype.numerical: table.columns[Stype.numerical],
            Stype.categorical: table.columns[Stype.categorical],
        },
        numerical=numerical,
        categorical=table.categorical,
    )


def _step_error(
    exc: Exception,
    phase: str,
    position: int,
    step: Processor,
) -> Exception:
    message = f"{phase} step {position} ({step.__class__.__name__}): {exc}"
    try:
        return exc.__class__(message)
    except Exception:  # noqa: BLE001 - not all exceptions rebuild from a message
        return RuntimeError(message)
