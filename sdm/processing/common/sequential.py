from collections.abc import Iterable
from typing import cast

import torch
from typing_extensions import Self

from sdm import Stype, TableTensor
from sdm.processing import InvertibleMixin, Processor
from sdm.processing._callable import ProcessorLike, _CallableProcessor


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

    def __init__(self, *args: ProcessorLike) -> None:
        super().__init__()
        self.extend(args)
        self.requires_fit = any(child.requires_fit for child in self)

    def append(self, processor: ProcessorLike) -> Self:
        r"""Append a processor to this sequence.

        Args:
            processor: The processor to append.
        """
        if isinstance(processor, Sequential):
            for child in processor.children():
                self.add_module(str(len(self)), child)
        elif isinstance(processor, Processor):
            self.add_module(str(len(self)), processor)
        elif callable(processor):
            self.add_module(str(len(self)), _CallableProcessor(processor))
        else:
            raise TypeError(
                f"Element must be a 'Processor' or callable "
                f"(got '{type(processor).__name__}')"
            )
        return self

    def extend(self, processors: Iterable[ProcessorLike]) -> Self:
        r"""Append multiple processors to this sequence.

        Args:
            processors: The processors to append.
        """
        for processor in processors:
            self.append(processor)
        return self

    def _fit(
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        raise NotImplementedError

    def fit(  # noqa: D102
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> "Sequential":
        out = table
        for i, child in enumerate(self):
            if i < len(self) - 1:
                out = child.fit_transform(out, generator=generator)
            else:
                child.fit(out, generator=generator)
        if self.requires_fit:
            self._fitted = True
        return self

    def _transform(self, table: TableTensor) -> TableTensor:
        out = table
        for child in self:
            out = child.transform(out)
        return out

    def fit_transform(  # noqa: D102
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> TableTensor:
        out = table
        for child in self:
            out = child.fit_transform(out, generator=generator)
        if self.requires_fit:
            self._fitted = True
        return out

    def _inverse_transform(self, table: TableTensor) -> TableTensor:
        out = table
        for child in reversed(list(self)):
            fn = getattr(child, "inverse_transform", None)
            if not callable(fn):
                raise AttributeError(
                    f"'{child.__class__.__name__}' object has no attribute "
                    f"'inverse_transform'"
                )
            out = fn(out)
        return out

    def __iter__(self) -> Iterator[Processor]:
        return cast(Iterator[Processor], self.children())

    def __len__(self) -> int:
        return len(self._modules)

    def __iadd__(self, other: ProcessorLike | Iterable[ProcessorLike]) -> Self:
        if isinstance(other, Processor) or callable(other):
            self.append(other)
        else:
            self.extend(other)
        return self

    def __repr__(self, *, indent: int = 0) -> str:
        if len(self) == 0:
            return super().__repr__(indent=indent)
        reprs = "".join(
            [f"{child.__repr__(indent=indent + 2)},\n" for child in self]
        )
        return (
            f"{' ' * indent}{self.__class__.__name__}(\n{reprs}{' ' * indent})"
        )
