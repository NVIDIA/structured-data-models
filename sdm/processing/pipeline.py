from collections.abc import Iterable, Iterator

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
        steps = tuple(steps or ())
        for step in steps:
            if not isinstance(step, Processor):
                raise TypeError(
                    f"Expected a Processor step "
                    f"(got '{step.__class__.__name__}')"
                )
        self.steps = steps

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
                    numerical = step.transform(numerical)
                    if not isinstance(numerical, Tensor):
                        raise TypeError(
                            "Expected the step to return a Tensor for the "
                            "numerical block (got "
                            f"'{type(numerical).__name__}')"
                        )
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
                numerical = step.transform(numerical)
                if not isinstance(numerical, Tensor):
                    raise TypeError(
                        "Expected the step to return a Tensor for the "
                        f"numerical block (got '{type(numerical).__name__}')"
                    )
            except Exception as exc:
                raise _step_error(exc, phase, position, step) from exc
        return _with_numerical(table, numerical)

    def fit_transform(self, table: TableTensor) -> TableTensor:
        """Fit and transform ``table`` by threading steps in order."""
        return self._fit_transform(table, phase="pipeline")

    def _fit_transform(self, table: TableTensor, *, phase: str) -> TableTensor:
        if len(self.steps) == 0:
            return table
        numerical = table.numerical
        for position, step in enumerate(self.steps):
            try:
                numerical = step.fit_transform(numerical)
                if not isinstance(numerical, Tensor):
                    raise TypeError(
                        "Expected the step to return a Tensor for the "
                        f"numerical block (got '{type(numerical).__name__}')"
                    )
            except Exception as exc:
                raise _step_error(exc, phase, position, step) from exc
        return _with_numerical(table, numerical)

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
                numerical = step.inverse_transform(numerical)
                if not isinstance(numerical, Tensor):
                    raise TypeError(
                        "Expected the step to return a Tensor for the "
                        f"numerical block (got '{type(numerical).__name__}')"
                    )
            except Exception as exc:
                raise _step_error(exc, phase, position, step) from exc
        return _with_numerical(table, numerical)

    def __len__(self) -> int:
        return len(self.steps)

    def __iter__(self) -> Iterator[Processor]:
        return iter(self.steps)

    def __repr__(self) -> str:
        steps = (
            " -> ".join(step.__class__.__name__ for step in self.steps)
            or "identity"
        )
        return f"{self.__class__.__name__}({steps})"


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
