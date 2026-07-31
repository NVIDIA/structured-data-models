from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from torch import Tensor

from sdm.processing.base import Processor
from sdm.stype import Stype
from sdm.tensor import StringTensor, TableTensor

if TYPE_CHECKING:
    pass


class LLMTextEmbed(Processor):
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
        embedding_model: torch.nn.Module,
        embedding_dim: int,
        *,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self._embedding_model = embedding_model
        self._embedding_dim = embedding_dim
        if dtype is not None and not dtype.is_floating_point:
            raise ValueError(f"`dtype` must be floating-point (got {dtype}).")
        self._dtype = dtype

    def _transform(self, table: TableTensor) -> TableTensor:
        device = table.text.device
        dtype = self._dtype or torch.get_default_dtype()

        text_names = table.columns[Stype.text]
        leading_shape = table.text.shape[:-1]
        embedder = self._embedding_model
        dim = _embedding_dim

        blocks: list[Tensor] = []
        names: list[str] = []
        for column, name in enumerate(text_names):
            column_text = cast(
                StringTensor,
                table.text[..., column].reshape(-1),
            )
            n_values = column_text.numel()
            if n_values == 0:
                block = torch.zeros(
                    (*leading_shape, dim),
                    dtype=dtype,
                    device=device,
                )
            else:
                strings = (
                    column_text.to_cudf()
                    if column_text.is_cuda
                    else column_text.to_arrow()
                )
                block = embedder(strings)
                if (
                    block.dim() != 2
                    or block.size(0) != n_values
                    or block.size(-1) != dim
                ):
                    raise ValueError(
                        f"Expected 'encode' to return a [{n_values}, {dim}] "
                        f"tensor (got {tuple(block.size())})"
                    )
                block = block.to(device=device, dtype=dtype).reshape(
                    *leading_shape,
                    dim,
                )
            blocks.append(block)
            names.extend(f"{name}_{i}" for i in range(dim))

        numerical = (
            torch.cat(blocks, dim=-1)
            if blocks
            else torch.zeros((*leading_shape, 0), dtype=dtype, device=device)
        )
        return table.__class__(
            columns={Stype.numerical: tuple(names)},
            numerical=numerical,
        )
