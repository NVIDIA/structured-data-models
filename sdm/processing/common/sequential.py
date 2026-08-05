from collections.abc import Iterable, Iterator
from typing import cast

import torch
from typing_extensions import Self

from sdm import Stype
from sdm.processing.base import Processor
from sdm.processing.ensemble import (
    EnsembleInvertibleMixin,
    EnsembleProcessor,
    EnsembleProcessorAdapter,
)
from sdm.tensor import EnsembleTable


class Sequential(EnsembleProcessor, EnsembleInvertibleMixin):
    r"""Apply processors and callables to a table in sequence.

    Each child consumes the previous child's output. Ordinary processors are
    adapted once when an ensemble is processed; ensemble processors are used
    directly.

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

    def _fit_ensemble(
        self,
        ensemble_table: EnsembleTable,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        # Only intermediate outputs are needed to fit the following child.
        out = ensemble_table
        children = []
        for name, child in tuple(self._modules.items()):
            if not isinstance(child, EnsembleProcessor):
                child = EnsembleProcessorAdapter(cast(Processor, child))
                self._modules[name] = child
            children.append(child)
        for child in children[:-1]:
            out = child.fit_transform_ensemble(out, generator=generator)
        if children:
            children[-1].fit_ensemble(out, generator=generator)

    def _fit_transform_ensemble(
        self,
        ensemble_table: EnsembleTable,
        *,
        generator: torch.Generator | None = None,
    ) -> EnsembleTable:
        out = ensemble_table
        for name, child in tuple(self._modules.items()):
            if not isinstance(child, EnsembleProcessor):
                child = EnsembleProcessorAdapter(cast(Processor, child))
                self._modules[name] = child
            out = child.fit_transform_ensemble(out, generator=generator)
        return out

    def _transform_ensemble(
        self,
        ensemble_table: EnsembleTable,
    ) -> EnsembleTable:
        out = ensemble_table
        for name, child in tuple(self._modules.items()):
            if not isinstance(child, EnsembleProcessor):
                child = EnsembleProcessorAdapter(cast(Processor, child))
                self._modules[name] = child
            out = child.transform_ensemble(out)
        return out

    def _inverse_transform_ensemble(
        self,
        ensemble_table: EnsembleTable,
    ) -> EnsembleTable:
        """Apply fitted child inverses in reverse order.

        Args:
            ensemble_table: Ensemble table in the transformed representation.

        Returns:
            Ensemble table restored to its representation before transform.
        """
        out = ensemble_table
        for name, child in reversed(tuple(self._modules.items())):
            if not isinstance(child, EnsembleProcessor):
                child = EnsembleProcessorAdapter(cast(Processor, child))
                self._modules[name] = child
            fn = getattr(child, "inverse_transform_ensemble", None)
            if not callable(fn):
                raise AttributeError(
                    f"{child.__class__.__name__!r} object has no "
                    "attribute 'inverse_transform_ensemble'"
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
