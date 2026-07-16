from collections.abc import Callable

import torch

from sdm.processing.base import InvertibleMixin, Processor
from sdm.stype import Stype
from sdm.tensor import TableTensor


class _CallableProcessor(Processor):
    """Adapt a stateless callable to the :class:`Processor` interface."""

    supported_stypes = frozenset(Stype)
    requires_fit = False

    def __init__(
        self,
        function: Callable[[TableTensor], TableTensor],
    ) -> None:
        super().__init__()
        self.function = function

    def _transform(self, table: TableTensor) -> TableTensor:
        return self.function(table)

    def __repr__(self, *, indent: int = 0) -> str:
        name = getattr(
            self.function,
            "__name__",
            self.function.__class__.__name__,
        )
        return f"{' ' * indent}{name}"


class Sequential(Processor, InvertibleMixin):
    r"""Apply processors and stateless callables in sequence.

    A ``generator`` passed to ``fit()`` or ``fit_transform()`` is passed on
    to every step.

    Args:
        args: Sequence of :class:`Processor` instances or callables. Each
            callable accepts and returns a :class:`~sdm.tensor.TableTensor`
            and is treated as a stateless, non-invertible processor.
    """

    supported_stypes = frozenset(Stype)

    def __init__(
        self,
        *args: Processor | Callable[[TableTensor], TableTensor],
    ) -> None:
        super().__init__()
        steps: list[Processor] = []
        for index, step in enumerate(args):
            if isinstance(step, Processor):
                steps.append(step)
            elif callable(step):
                steps.append(_CallableProcessor(step))
            else:
                raise TypeError(
                    f"Sequential step {index} must be a Processor or "
                    f"callable, got {step.__class__.__name__}."
                )

        self.steps = tuple(steps)
        for i, step in enumerate(self.steps):
            self.add_module(str(i), step)
        self.requires_fit = any(step.requires_fit for step in self.steps)

    def _fit(
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        out = table
        for step in self.steps:
            out = step.fit_transform(out, generator=generator)

    def fit(  # noqa: D102
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> "Sequential":
        out = table
        for step in self.steps:
            out = step.fit_transform(out, generator=generator)
        if self.requires_fit:
            self._fitted = True
        return self

    def _transform(self, table: TableTensor) -> TableTensor:
        out = table
        for step in self.steps:
            out = step.transform(out)
        return out

    def fit_transform(  # noqa: D102
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> TableTensor:
        out = table
        for step in self.steps:
            out = step.fit_transform(out, generator=generator)
        if self.requires_fit:
            self._fitted = True
        return out

    def _inverse_transform(self, table: TableTensor) -> TableTensor:
        out = table
        for step in self.steps[::-1]:
            fn = getattr(step, "inverse_transform", None)
            if not callable(fn):
                raise AttributeError(
                    f"'{step.__class__.__name__}' object has no attribute "
                    f"'inverse_transform"
                )
            out = fn(out)
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
