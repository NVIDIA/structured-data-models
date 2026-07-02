"""Ordered execution of processing steps over table data."""

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

    # TODO: Consider whether Pipeline should itself be a Processor (Composite
    # pattern) so pipelines can nest and be used anywhere a step is expected.
    # Blocked on the current Processor contract: it is Tensor-typed
    # (forward/fit are Tensor -> Tensor) and an nn.Module, whereas Pipeline is
    # TableTensor -> TableTensor and holds steps as a plain tuple. Revisit once
    # the routing contract below is unified to TableTensor -> TableTensor with
    # capability declarations; until then, prefer a shared structural protocol
    # (fit/transform/inverse_transform) over direct inheritance.

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
        """Fit steps in order.

        Block-scoped steps fit on the pipeline-selected tensor block.
        Table steps fit on the full table and may update the table threaded
        to later steps.

        Args:
            table: Data whose last dimension is the column dimension.

        Returns:
            The pipeline itself, to allow call chaining.
        """
        current = table
        last = len(self.steps) - 1
        for position, step in enumerate(self.steps):
            try:
                step.fit(_step_input(step, current))
                if position != last:
                    current = _transform_step(step, current)
            except Exception as exc:
                raise _step_error(exc, position, step) from exc
        return self

    def transform(self, table: TableTensor) -> TableTensor:
        """Transform ``table`` by threading steps in order.

        Args:
            table: Data whose last dimension is the column dimension.

        Returns:
            A table with each step applied; the input is returned unchanged
            when the pipeline is empty.
        """
        current = table
        for position, step in enumerate(self.steps):
            try:
                current = _transform_step(step, current)
            except Exception as exc:
                raise _step_error(exc, position, step) from exc
        return current

    def fit_transform(self, table: TableTensor) -> TableTensor:
        """Fit and transform ``table`` by threading steps in order.

        Args:
            table: Data whose last dimension is the column dimension.

        Returns:
            A table with each fitted step applied; the input is returned
            unchanged when the pipeline is empty.
        """
        current = table
        for position, step in enumerate(self.steps):
            try:
                step.fit(_step_input(step, current))
                current = _transform_step(step, current)
            except Exception as exc:
                raise _step_error(exc, position, step) from exc
        return current

    def inverse_transform(self, table: TableTensor) -> TableTensor:
        """Apply invertible steps in reverse order to ``table``.

        Args:
            table: Data whose last dimension is the column dimension. Every
                step must mix in :class:`~sdm.processing.InvertibleMixin`.

        Returns:
            A table with invertible steps reversed; the input is returned
            unchanged when the pipeline is empty.
        """
        current = table
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
                current = _transform_step(step, current, inverse=True)
            except Exception as exc:
                raise _step_error(exc, position, step) from exc
        return current

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


def _step_input(step: Processor, table: TableTensor) -> Tensor:
    scope = _step_scope(step)
    if scope == "table":
        return table
    return table.numerical


def _transform_step(
    step: Processor,
    table: TableTensor,
    *,
    inverse: bool = False,
) -> TableTensor:
    scope = _step_scope(step)

    if scope == "table":
        if inverse:
            if not isinstance(step, InvertibleMixin):
                raise TypeError(
                    "Expected invertible step for inverse_transform"
                )
            output = step.inverse_transform(table)
        else:
            output = step.transform(table)
        if not isinstance(output, TableTensor):
            raise TypeError(
                "Expected the table step to return a TableTensor "
                f"(got '{type(output).__name__}')"
            )
        return output

    if inverse:
        if not isinstance(step, InvertibleMixin):
            raise TypeError("Expected invertible step for inverse_transform")
        output = step.inverse_transform(table.numerical)
    else:
        output = step.transform(table.numerical)
    if not isinstance(output, Tensor):
        raise TypeError(
            "Expected the block-scoped step to return a Tensor "
            f"(got '{type(output).__name__}')"
        )
    return _with_default_block(table, output)


def _step_scope(step: Processor) -> str:
    # TODO: Treat input_scope as routing granularity only, not a semantic-type
    # capability declaration. Future StypeDispatch should choose which stype
    # block is passed to a block-scoped processor, even when that processor can
    # support multiple stypes.
    scope = step.input_scope
    if scope not in {"block", "table"}:
        raise ValueError(
            "Expected processor input_scope to be 'block' or 'table' "
            f"(got '{scope}')"
        )
    return scope


def _with_default_block(table: TableTensor, numerical: Tensor) -> TableTensor:
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
