from sdm.processing import InvertibleMixin, Processor
from sdm.stype import Stype
from sdm.tensor import TableTensor


class Sequential(Processor, InvertibleMixin):
    r"""Apply a list of :class:`Processor` instances in sequence.

    Args:
        args: Sequence of :class:`Processor` instances.
    """

    #: The semantic types accepted by the first :class:`Processor` in the
    #: sequence.
    supported_stypes: frozenset[Stype]
    #: Whether any :class:`Processor` in the sequence requires fitting.
    requires_fit: bool

    def __init__(self, *args: Processor) -> None:
        super().__init__()
        self.steps: tuple[Processor, ...] = args
        if len(self.steps) > 0:
            self.supported_stypes = self.steps[0].supported_stypes
        else:
            self.supported_stypes = frozenset(Stype) - {Stype.id}
        self.requires_fit = any(step.requires_fit for step in self.steps)

    def _fit(self, input: TableTensor) -> None:
        out = input
        for step in self.steps[:-1]:
            out = step.fit_transform(out)
        if len(self.steps) > 0:
            self.steps[-1].fit(out)

    def _transform(self, input: TableTensor) -> TableTensor:
        out = input
        for step in self.steps:
            out = step.transform(out)
        return out

    def fit_transform(self, input: TableTensor) -> TableTensor:
        r"""Fit the processor and transform ``input``.

        :meta private:
        """
        out = input
        for step in self.steps:
            out = step.fit_transform(out)
        if self.requires_fit:
            self._fitted = True
        return out

    def _inverse_transform(self, input: TableTensor) -> TableTensor:
        out = input
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
