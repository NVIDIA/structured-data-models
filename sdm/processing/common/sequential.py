from collections.abc import Iterable, Iterator
from typing import cast

import torch
from typing_extensions import Self

from sdm import Stype
from sdm.processing.base import InvertibleMixin, Processor
from sdm.processing.ensemble import (
    EnsembleFitContext,
    EnsembleProcessor,
    as_ensemble_processor,
)
from sdm.tensor import EnsembleTable, TableTensor


class Sequential(EnsembleProcessor, InvertibleMixin):
    r"""Apply processors and callables in sequence.

    Args:
        args: Sequence of :class:`Processor` instances or callables.
    """

    supported_stypes = frozenset(Stype)

    def __init__(self, *args: object) -> None:
        super().__init__()
        self.requires_fit = False
        self.extend(args)

    def append(self, processor: object) -> Self:
        r"""Append a processor or callable to this sequence.

        Args:
            processor: The processor to append.
        """
        processor = Processor.as_processor(processor)
        if isinstance(processor, Sequential):
            for child in processor.children():
                self.add_module(str(len(self)), child)
        else:
            self.add_module(str(len(self)), processor)

        self.requires_fit = self.requires_fit or processor.requires_fit
        self._fitted = False
        return self

    def extend(self, processors: Iterable[object]) -> Self:
        r"""Append multiple processors or callables to this sequence.

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
        out = table
        for i, child in enumerate(self):
            if i < len(self) - 1:
                out = child.fit_transform(out, generator=generator)
            else:
                child.fit(out, generator=generator)

    def _transform(self, table: TableTensor) -> TableTensor:
        out = table
        for child in self:
            out = child.transform(out)
        return out

    def _fit_transform(
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> TableTensor:
        out = table
        for child in self:
            out = child.fit_transform(out, generator=generator)
        return out

    def fit_transform_ensemble(
        self,
        table: EnsembleTable,
        *,
        context: EnsembleFitContext,
    ) -> EnsembleTable:
        r"""Fit and transform all children in Recipe order."""
        for name, child in tuple(self._modules.items()):
            self._modules[name] = as_ensemble_processor(cast(Processor, child))

        out = table
        for index, child in enumerate(self):
            out = cast(EnsembleProcessor, child).fit_transform_ensemble(
                out,
                context=context.child(str(index)),
            )
        return out

    def transform_ensemble(self, table: EnsembleTable) -> EnsembleTable:
        r"""Transform an ensemble through all fitted children."""
        out = table
        for child in self:
            out = cast(EnsembleProcessor, child).transform_ensemble(out)
        return out

    def inverse_transform_ensemble(
        self,
        table: EnsembleTable,
    ) -> EnsembleTable:
        r"""Apply fitted child inverses in reverse order."""
        out = table
        for child in reversed(list(self)):
            out = cast(
                EnsembleProcessor,
                child,
            ).inverse_transform_ensemble(out)
        return out

    def _inverse_transform(self, table: TableTensor) -> TableTensor:
        out = table
        for child in reversed(list(self)):
            fn = getattr(child, "inverse_transform", None)
            if not callable(fn):
                raise AttributeError(
                    f"{child.__class__.__name__!r} object has no attribute "
                    f"'inverse_transform'"
                )
            out = fn(out)
        return out

    def __iter__(self) -> Iterator[Processor]:
        return cast(Iterator[Processor], self.children())

    def __len__(self) -> int:
        return len(self._modules)

    def __iadd__(self, other: object) -> Self:
        self.append(Processor.as_processor(other))
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
