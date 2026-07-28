from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, cast, runtime_checkable

import torch
from torch import Tensor

from sdm.processing.base import Processor
from sdm.stype import Stype
from sdm.tensor import StringTensor, TableTensor

if TYPE_CHECKING:
    import cudf
    import pyarrow as pa


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

    def encode(self, strings: cudf.Series | pa.Array) -> Tensor:
        """Embed strings into a ``[len(strings), dim]`` tensor.

        Args:
            strings: cuDF Series or PyArrow Array of strings.
        """
        ...


class LLMTransformer(Processor):
    r"""Encode text columns with a user-provided embedding model.

    Each text column is embedded cell-by-cell through ``embedder`` and
    expands to a block of ``embedder.dim`` numerical features; blocks are
    concatenated in column order into the numerical output.

    Args:
        embedder: Pre-loaded :class:`Embedder` mapping a batch of strings to
            a ``[n, dim]`` embedding tensor.
        dtype: Floating-point dtype of the returned numerical features. If
            ``None``, uses :func:`torch.get_default_dtype`.
    """

    requires_fit = False
    supported_stypes = frozenset({Stype.text})

    def __init__(
        self,
        embedder: Embedder,
        *,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self._embedder = embedder
        self._dtype = dtype

    @property
    def embedder(self) -> Embedder:
        """The wrapped embedding model."""
        return self._embedder

    def _transform(self, table: TableTensor) -> TableTensor:
        device = table.text.device
        dtype = self._dtype or torch.get_default_dtype()

        text_names = table.columns[Stype.text]
        n_rows = table.text.size(0)
        embedder = self.embedder
        dim = embedder.dim

        blocks: list[Tensor] = []
        names: list[str] = []
        for column, name in enumerate(text_names):
            if n_rows == 0:
                block = torch.zeros((0, dim), dtype=dtype, device=device)
            else:
                column_text = cast(StringTensor, table.text[:, column])
                strings = (
                    column_text.to_cudf()
                    if column_text.is_cuda
                    else column_text.to_arrow()
                )
                block = embedder.encode(strings)
                if (
                    block.dim() != 2
                    or block.size(0) != n_rows
                    or block.size(-1) != dim
                ):
                    raise ValueError(
                        f"Expected 'encode' to return a [{n_rows}, {dim}] "
                        f"tensor (got {tuple(block.size())})"
                    )
                block = block.to(device=device, dtype=dtype)
            blocks.append(block)
            names.extend(f"{name}_{i}" for i in range(dim))

        numerical = (
            torch.cat(blocks, dim=-1)
            if blocks
            else torch.zeros((n_rows, 0), dtype=dtype, device=device)
        )
        return table.__class__(
            columns={Stype.numerical: tuple(names)},
            numerical=numerical,
        )
