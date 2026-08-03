from collections.abc import Iterable, Iterator
from typing import cast

import torch
from typing_extensions import Self

from sdm import EnsembleTable, Stype, TableTensor
from sdm.processing.base import Processor
from sdm.processing.ensemble import (
    EnsembleInvertibleMixin,
    EnsembleProcessor,
    EnsembleProcessorAdapter,
)


class Sequential(EnsembleProcessor, EnsembleInvertibleMixin):
    r"""Apply processors and callables in sequence.

    Args:
        args: Sequence of :class:`Processor` instances or callables.
    """

    supported_stypes = frozenset(Stype)

    def __init__(self, *args: object) -> None:
        super().__init__()
        self.extend(args)
        self.requires_fit = any(child.requires_fit for child in self)

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

    def _fit_transform_ensemble(
        self,
        table: EnsembleTable,
        *,
        generator: torch.Generator | None = None,
    ) -> EnsembleTable:
        out = table
        for name, child in tuple(self._modules.items()):
            processor = EnsembleProcessorAdapter.adapt(cast(Processor, child))
            if processor is not child:
                self._modules[name] = processor
            out = processor.fit_transform_ensemble(out, generator=generator)
        return out

    def _transform_ensemble(self, table: EnsembleTable) -> EnsembleTable:
        out = table
        for name, child in tuple(self._modules.items()):
            processor = EnsembleProcessorAdapter.adapt(cast(Processor, child))
            if processor is not child:
                self._modules[name] = processor
            out = processor.transform_ensemble(out)
        return out

    def _inverse_transform_ensemble(
        self,
        table: EnsembleTable,
    ) -> EnsembleTable:
        """Apply fitted child inverses in reverse order.

        Args:
            table: Ensemble table in the transformed representation.

        Returns:
            Ensemble table restored to its representation before transform.
        """
        out = table
        for name, child in reversed(tuple(self._modules.items())):
            processor = EnsembleProcessorAdapter.adapt(cast(Processor, child))
            if processor is not child:
                self._modules[name] = processor
            fn = getattr(processor, "inverse_transform_ensemble", None)
            if not callable(fn):
                raise AttributeError(
                    f"{processor.__class__.__name__!r} object has no "
                    "attribute 'inverse_transform_ensemble'"
                )
            out = fn(out)
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
