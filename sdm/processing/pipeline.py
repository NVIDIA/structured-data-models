"""Ordered execution of processing steps over a table's numerical block."""

from collections.abc import Iterable, Iterator

from torch import Tensor
from typing_extensions import Self

from sdm import Stype
from sdm.processing.base import InvertibleMixin, Processor
from sdm.tensor import TableTensor


class Pipeline:
    """Ordered sequence of processing steps applied to one data role.

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
        """Fit steps in order using the numerical block of ``table``.

        Args:
            table: Data whose numerical block ``[..., C_num]`` fits the steps;
                categorical columns are ignored.

        Returns:
            The pipeline itself, to allow call chaining.
        """
        numerical = table.numerical
        last = len(self.steps) - 1
        for position, step in enumerate(self.steps):
            try:
                step.fit(numerical)
                if position != last:
                    numerical = step.transform(numerical)
            except Exception as exc:
                raise _step_error(exc, position, step) from exc
        return self

    def transform(self, table: TableTensor) -> TableTensor:
        """Transform ``table`` by applying steps to its numerical block.

        Args:
            table: Data with a numerical block ``[..., C_num]`` to transform;
                categorical columns pass through unchanged.

        Returns:
            A table with the transformed numerical block; the input is
            returned unchanged when the pipeline is empty.
        """
        if len(self.steps) == 0:
            return table
        numerical = table.numerical
        for position, step in enumerate(self.steps):
            try:
                numerical = step.transform(numerical)
            except Exception as exc:
                raise _step_error(exc, position, step) from exc
        return _with_numerical(table, numerical)

    def fit_transform(self, table: TableTensor) -> TableTensor:
        """Fit and transform ``table`` by threading steps in order.

        Args:
            table: Data with a numerical block ``[..., C_num]`` used to both
                fit and transform the steps; categorical columns pass through
                unchanged.

        Returns:
            A table with the transformed numerical block; the input is
            returned unchanged when the pipeline is empty.
        """
        if len(self.steps) == 0:
            return table
        numerical = table.numerical
        for position, step in enumerate(self.steps):
            try:
                numerical = step.fit_transform(numerical)
            except Exception as exc:
                raise _step_error(exc, position, step) from exc
        return _with_numerical(table, numerical)

    def inverse_transform(self, table: TableTensor) -> TableTensor:
        """Apply invertible steps in reverse order to ``table``.

        Args:
            table: Data with a numerical block ``[..., C_num]`` to invert;
                categorical columns pass through unchanged. Every step must
                mix in :class:`~sdm.processing.InvertibleMixin`.

        Returns:
            A table with the inverted numerical block; the input is returned
            unchanged when the pipeline is empty.
        """
        if len(self.steps) == 0:
            return table
        numerical = table.numerical
        for position, step in reversed(tuple(enumerate(self.steps))):
            if not isinstance(step, InvertibleMixin):
                raise _step_error(
                    TypeError(
                        "Expected invertible step for inverse_transform"
                    ),
                    position,
                    step,
                )
            try:
                numerical = step.inverse_transform(numerical)
            except Exception as exc:
                raise _step_error(exc, position, step) from exc
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
    position: int,
    step: Processor,
) -> Exception:
    message = f"step {position} ({step.__class__.__name__}): {exc}"
    try:
        return exc.__class__(message)
    except Exception:  # noqa: BLE001 - not all exceptions rebuild from a message
        return RuntimeError(message)
