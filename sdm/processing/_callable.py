import ast
import inspect
import textwrap
from collections.abc import Callable
from typing import TypeAlias

from sdm.processing.base import Processor
from sdm.stype import Stype
from sdm.tensor import TableTensor

ProcessorCallable: TypeAlias = Callable[[TableTensor], TableTensor]
ProcessorLike: TypeAlias = Processor | ProcessorCallable


class _CallableProcessor(Processor):
    """Adapt a stateless callable to the :class:`Processor` interface."""

    supported_stypes = frozenset(Stype)
    requires_fit = False

    def __init__(self, function: ProcessorCallable) -> None:
        super().__init__()
        self.function = function
        self._display_name = getattr(
            function,
            "__name__",
            function.__class__.__name__,
        )
        if self._display_name == "<lambda>":
            self._display_name = "lambda"
            try:
                source = textwrap.dedent(inspect.getsource(function))
                tree = ast.parse(source)
            except (OSError, TypeError, IndentationError, SyntaxError):
                pass
            else:
                lambdas = [
                    node
                    for node in ast.walk(tree)
                    if isinstance(node, ast.Lambda)
                ]
                if len(lambdas) == 1:
                    self._display_name = " ".join(
                        ast.unparse(lambdas[0]).split()
                    )
                    if len(self._display_name) > 72:
                        self._display_name = (
                            f"{self._display_name[:69].rstrip()}..."
                        )

    def _transform(self, table: TableTensor) -> TableTensor:
        return self.function(table)

    def __repr__(self, *, indent: int = 0) -> str:
        return f"{' ' * indent}{self._display_name}"


def as_processor(step: ProcessorLike, *, label: str) -> Processor:
    """Normalize a processor or stateless callable to a processor."""
    if isinstance(step, Processor):
        return step
    if callable(step):
        return _CallableProcessor(step)
    raise TypeError(
        f"{label} must be a Processor or callable, got "
        f"{step.__class__.__name__}."
    )
