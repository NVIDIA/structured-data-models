from sdm.processing.base import InvertibleMixin, Processor
from sdm.stype import Stype
from sdm.tensor import TableTensor


class Sequential(Processor, InvertibleMixin):
    r"""Apply a number of :class:`Processor` instances in sequence.

    Args:
        args: Sequence of :class:`Processor` instances.
    """

    supported_stypes = frozenset(Stype)

    def __init__(self, *args: Processor) -> None:
        super().__init__()
        self.steps: tuple[Processor, ...] = args
        for i, step in enumerate(self.steps):
            self.add_module(str(i), step)
        self.requires_fit = any(step.requires_fit for step in self.steps)

    def _fit(self, inp: TableTensor) -> None:
        out = inp
        for step in self.steps:
            out = step.fit_transform(out)

    def fit(self, inp: TableTensor) -> "Sequential":  # noqa: D102
        out = inp
        for step in self.steps:
            out = step.fit_transform(out)
        if self.requires_fit:
            self._fitted = True
        return self

    def _transform(self, inp: TableTensor) -> TableTensor:
        out = inp
        for step in self.steps:
            out = step.transform(out)
        return out

    def fit_transform(self, inp: TableTensor) -> TableTensor:  # noqa: D102
        out = inp
        for step in self.steps:
            out = step.fit_transform(out)
        if self.requires_fit:
            self._fitted = True
        return out

    def _inverse_transform(self, inp: TableTensor) -> TableTensor:
        out = inp
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
