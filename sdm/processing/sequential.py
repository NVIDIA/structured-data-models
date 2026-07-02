from torch import Tensor

from sdm.processing.base import InvertibleMixin, Processor


class Sequential(Processor, InvertibleMixin):
    r"""Apply a number of :class:`Processor` instances in sequence.

    Args:
        args: Sequence of :class:`Processor` instances.
    """

    def __init__(self, *args: Processor) -> None:
        self.steps: tuple[Processor, ...] = args
        self.requires_fit = any(step.requires_fit for step in self.steps)

    def _fit(self, input: Tensor) -> None:
        self.fit_transform(input)

    def transform(self, input: Tensor) -> Tensor:  # noqa: D102
        out = input
        for step in self.steps:
            out = step.transform(out)
        return out

    def fit_transform(self, input: Tensor) -> Tensor:  # noqa: D102
        out = input
        for step in self.steps:
            out = step.fit_transform(out)
        return out

    def inverse_transform(self, input: Tensor) -> Tensor:  # noqa: D102
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
