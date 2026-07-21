from typing import Protocol, cast, runtime_checkable

import torch
from torch import Tensor

from sdm.processing.base import Processor, SharedState
from sdm.stype import Stype
from sdm.tensor import StringTensor, TableTensor


@runtime_checkable
class Embedder(Protocol):
    """Anything that maps a batch of strings to one embedding per string.

    Implementations own batching, truncation, pooling, and where the model
    runs (local module or remote endpoint); the processor only relies on
    this interface.
    """

    @property
    def dim(self) -> int:
        """Width of the returned embeddings."""
        ...

    def encode(self, strings: list[str]) -> Tensor:
        """Embed each string into a ``[len(strings), dim]`` tensor."""
        ...


class LLMEncoder(Processor):
    r"""Encode text columns with a user-provided embedding model.

    Each text column is embedded cell-by-cell through ``embedder`` and
    expands to a block of ``embedder.dim`` numerical features; blocks are
    concatenated in column order into the numerical output. The embedder is
    pre-loaded by the caller and only referenced here (never copied), so
    ensemble members share a single model instance.

    Args:
        embedder: Pre-loaded :class:`Embedder` mapping a batch of strings to
            a ``[n, dim]`` embedding tensor.
    """

    requires_fit = False
    supported_stypes = frozenset({Stype.text})

    def __init__(self, embedder: Embedder) -> None:
        super().__init__()
        self._embedder = SharedState(embedder)

    @property
    def embedder(self) -> Embedder:
        """The wrapped embedding model."""
        return self._embedder.value

    def _transform(self, table: TableTensor) -> TableTensor:
        device = table.numerical.device
        dtype = table.numerical.dtype
        if not dtype.is_floating_point:
            dtype = torch.get_default_dtype()
        text_names = table.columns[Stype.text]
        n_rows = table.text.size(0)
        dim = self.embedder.dim

        blocks: list[Tensor] = []
        names: list[str] = []
        for column, name in enumerate(text_names):
            if n_rows == 0:
                block = torch.zeros((0, dim), dtype=dtype, device=device)
            else:
                column_text = cast(StringTensor, table.text[:, column])
                strings = column_text.to_arrow().to_pylist()
                block = self.embedder.encode(strings)
                if block.dim() != 2 or block.size(0) != n_rows:
                    raise ValueError(
                        f"Expected 'encode' to return a [{n_rows}, dim] "
                        f"tensor (got {tuple(block.size())})"
                    )
                block = block.to(device=device, dtype=dtype)
            blocks.append(block)
            names.extend(f"{name}_{i}" for i in range(block.size(-1)))

        numerical = (
            torch.cat(blocks, dim=-1)
            if blocks
            else torch.zeros((n_rows, 0), dtype=dtype, device=device)
        )
        return table.__class__(
            columns={Stype.numerical: tuple(names)},
            numerical=numerical,
        )
