from torch import Tensor

from sdm import Stype
from sdm.processing.base import InvertibleMixin, Processor
from sdm.tensor import TableTensor


class Sequential(Processor, InvertibleMixin):
    r"""Apply a number of :class:`Processor` instances in sequence.

    Args:
        args: Sequence of :class:`Processor` instances.
    """

    input_scope = "table"

    def __init__(self, *args: Processor) -> None:
        super().__init__()
        for step in args:
            if not isinstance(step, Processor):
                raise TypeError(
                    f"Expected a Processor step "
                    f"(got '{step.__class__.__name__}')"
                )
        self.steps: tuple[Processor, ...] = args
        self.requires_fit = any(step.requires_fit for step in self.steps)

    def _fit(self, input: Tensor) -> None:
        out = input
        for step in self.steps:
            step.fit(_step_input(step, out))
            out = _transform_step(step, out)

    def _transform(self, input: Tensor) -> Tensor:
        out = input
        for position, step in enumerate(self.steps):
            try:
                out = _transform_step(step, out)
            except Exception as exc:
                raise _step_error(exc, position, step) from exc
        return out

    def fit_transform(self, input: Tensor) -> Tensor:  # noqa: D102
        out = input
        for position, step in enumerate(self.steps):
            try:
                step.fit(_step_input(step, out))
                out = _transform_step(step, out)
            except Exception as exc:
                raise _step_error(exc, position, step) from exc
        if self.requires_fit:
            self._fitted = True
        return out

    def _inverse_transform(self, input: Tensor) -> Tensor:
        out = input
        for position, step in reversed(tuple(enumerate(self.steps))):
            try:
                out = _transform_step(step, out, inverse=True)
            except Exception as exc:
                raise _step_error(exc, position, step) from exc
        return out

    def __repr__(self, *, indent: int = 0) -> str:
        if len(self.steps) == 0:
            return super().__repr__(indent=indent)
        reprs = ",\n".join(
            [step.__repr__(indent=indent + 2) for step in self.steps]
        )
        return (
            f"{' ' * indent}{self.__class__.__name__}(\n"
            f"{reprs},\n"
            f"{' ' * indent})"
        )


def _step_input(step: Processor, input: Tensor) -> Tensor:
    scope = _step_scope(step)
    if scope == "table" or not isinstance(input, TableTensor):
        return input
    return input.numerical


def _transform_step(
    step: Processor,
    input: Tensor,
    *,
    inverse: bool = False,
) -> Tensor:
    scope = _step_scope(step)

    if scope == "table" or not isinstance(input, TableTensor):
        if inverse:
            fn = getattr(step, "inverse_transform", None)
            if not callable(fn):
                raise AttributeError(
                    f"'{step.__class__.__name__}' object has no attribute "
                    f"'inverse_transform"
                )
            output = fn(input)
        else:
            output = step.transform(input)
        if (
            scope == "table"
            and isinstance(input, TableTensor)
            and not isinstance(output, TableTensor)
        ):
            raise TypeError(
                "Expected the table step to return a TableTensor "
                f"(got '{type(output).__name__}')"
            )
        return output

    if inverse:
        fn = getattr(step, "inverse_transform", None)
        if not callable(fn):
            raise AttributeError(
                f"'{step.__class__.__name__}' object has no attribute "
                f"'inverse_transform"
            )
        output = fn(input.numerical)
    else:
        output = step.transform(input.numerical)

    if not isinstance(output, Tensor):
        raise TypeError(
            "Expected the block-scoped step to return a Tensor "
            f"(got '{type(output).__name__}')"
        )
    return _with_default_block(input, output)


def _step_scope(step: Processor) -> str:
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
