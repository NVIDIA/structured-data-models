from collections.abc import Iterable, Iterator
from typing import cast

import torch
from typing_extensions import Self

from sdm import Stype, StypeLike, TableTensor
from sdm.processing import InvertibleMixin, Processor


class Sequential(Processor, InvertibleMixin):
    r"""Apply processors and callables in sequence.

    Args:
        args: Sequence of :class:`Processor` instances or callables.
        passthrough_stypes: Semantic types passed unchanged around children
            that do not include them in ``supported_stypes``. Children that
            support a configured semantic type receive it normally.
    """

    supported_stypes = frozenset(Stype)

    def __init__(
        self,
        *args: object,
        passthrough_stypes: Iterable[StypeLike] = (),
    ) -> None:
        super().__init__()
        self.passthrough_stypes = frozenset(
            Stype(stype) for stype in passthrough_stypes
        )
        self.extend(args)
        self.requires_fit = any(child.requires_fit for child in self)

    def append(self, processor: object) -> Self:
        r"""Append a processor or callable to this sequence.

        Args:
            processor: The processor to append.
        """
        processor = Processor.as_processor(processor)
        if (
            isinstance(processor, Sequential)
            and len(processor.passthrough_stypes) == 0
        ):
            for child in processor.children():
                self.add_module(str(len(self)), child)
        else:
            self.add_module(str(len(self)), processor)

        self.requires_fit = any(child.requires_fit for child in self)
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

    def _split_passthrough(
        self,
        table: TableTensor,
        child: Processor,
    ) -> tuple[TableTensor, TableTensor | None]:
        stypes = tuple(
            stype
            for stype, columns in table.columns.items()
            if stype in self.passthrough_stypes
            and stype not in child.supported_stypes
            and len(columns) > 0
        )
        if len(stypes) == 0:
            return table, None
        return table.drop_stypes(stypes), table.select_stypes(stypes)

    @staticmethod
    def _restore_passthrough(
        table: TableTensor,
        passthrough: TableTensor | None,
    ) -> TableTensor:
        if passthrough is None:
            return table
        return cast(TableTensor, torch.cat((table, passthrough), dim=-1))

    def _fit(
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        out = table
        for i, child in enumerate(self):
            child_input, passthrough = self._split_passthrough(out, child)
            if i < len(self) - 1:
                out = child.fit_transform(
                    child_input,
                    generator=generator,
                )
                out = self._restore_passthrough(out, passthrough)
            else:
                child.fit(child_input, generator=generator)

    def _transform(self, table: TableTensor) -> TableTensor:
        out = table
        for child in self:
            child_input, passthrough = self._split_passthrough(out, child)
            out = child.transform(child_input)
            out = self._restore_passthrough(out, passthrough)
        return out

    def _fit_transform(
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> TableTensor:
        out = table
        for child in self:
            child_input, passthrough = self._split_passthrough(out, child)
            out = child.fit_transform(
                child_input,
                generator=generator,
            )
            out = self._restore_passthrough(out, passthrough)
        return out

    def _inverse_transform(self, table: TableTensor) -> TableTensor:
        out = table
        for child in reversed(list(self)):
            child_input, passthrough = self._split_passthrough(out, child)
            fn = getattr(child, "inverse_transform", None)
            if not callable(fn):
                raise AttributeError(
                    f"{child.__class__.__name__!r} object has no attribute "
                    f"'inverse_transform'"
                )
            out = fn(child_input)
            out = self._restore_passthrough(out, passthrough)
        return out

    def __iter__(self) -> Iterator[Processor]:
        return cast(Iterator[Processor], self.children())

    def __len__(self) -> int:
        return len(self._modules)

    def __iadd__(self, other: object) -> Self:
        self.append(Processor.as_processor(other))
        return self

    def __repr__(self, *, indent: int = 0) -> str:
        passthrough_repr = (
            "{"
            + ", ".join(
                repr(stype.value)
                for stype in Stype
                if stype in self.passthrough_stypes
            )
            + "}"
        )
        if len(self) == 0:
            if len(self.passthrough_stypes) > 0:
                return (
                    f"{' ' * indent}{self.__class__.__name__}("
                    f"passthrough_stypes={passthrough_repr})"
                )
            return super().__repr__(indent=indent)
        reprs = "".join(
            [f"{child.__repr__(indent=indent + 2)},\n" for child in self]
        )
        if len(self.passthrough_stypes) > 0:
            reprs += (
                f"{' ' * (indent + 2)}passthrough_stypes={passthrough_repr},\n"
            )
        return (
            f"{' ' * indent}{self.__class__.__name__}(\n{reprs}{' ' * indent})"
        )
