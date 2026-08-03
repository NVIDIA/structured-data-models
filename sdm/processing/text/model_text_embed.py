from __future__ import annotations

from typing import Any, cast

import torch
from torch import Tensor

from sdm.processing.base import Processor
from sdm.stype import Stype
from sdm.tensor import StringTensor, TableTensor


class ModelTextEmbed(Processor):
    r"""Embed text columns with a user-provided embedding model.

    Each text column is embedded cell-by-cell through ``embedding_model``.
    The model must return one embedding per text value as a
    :class:`torch.Tensor` with shape ``[n, embedding_dim]``. Embedding
    blocks are concatenated in column order into the numerical output.

    Args:
        embedding_model: Pre-loaded model called on each flattened text column.
            It must return a :class:`torch.Tensor` with shape
            ``[n, embedding_dim]``.
        embedding_dim: Width of each returned embedding.
    """

    requires_fit = False
    supported_stypes = frozenset({Stype.text})

    def __init__(
        self,
        embedding_model: torch.nn.Module,
        embedding_dim: int,
    ) -> None:
        super().__init__()
        self._embedding_model: torch.nn.Module = embedding_model
        self._embedding_dim: int = embedding_dim

    def __deepcopy__(self, memo: dict[int, Any]) -> ModelTextEmbed:
        copied = type(self)(
            embedding_model=self._embedding_model,
            embedding_dim=self._embedding_dim,
        )
        memo[id(self)] = copied
        copied._fitted = self._fitted
        copied.training = self.training
        return copied

    def _transform(self, table: TableTensor) -> TableTensor:
        device = table.device
        dtype = torch.get_default_dtype()

        text_names = table.columns[Stype.text]
        leading_shape = table.text.shape[:-1]
        embedding_model = self._embedding_model
        embedding_dim = self._embedding_dim

        blocks: list[Tensor] = []
        names: list[str] = []
        for col_idx, col_name in enumerate(text_names):
            column_text = cast(
                StringTensor,
                table.text[..., col_idx].reshape(-1),
            )
            n_values = column_text.numel()
            if n_values == 0:
                block = torch.zeros(
                    (*leading_shape, embedding_dim),
                    dtype=dtype,
                    device=device,
                )
            else:
                strings = (
                    column_text.to_cudf()
                    if column_text.is_cuda
                    else column_text.to_arrow()
                )
                block = embedding_model(strings)
                if (
                    block.dim() != 2
                    or block.size(0) != n_values
                    or block.size(-1) != embedding_dim
                ):
                    raise ValueError(
                        f"Expected 'embedding_model' to return a "
                        f"[{n_values}, {embedding_dim}] tensor "
                        f"(got {tuple(block.size())})"
                    )
                block = block.to(device=device, dtype=dtype).reshape(
                    *leading_shape,
                    embedding_dim,
                )
            blocks.append(block)
            names.extend(f"{col_name}_{i}" for i in range(embedding_dim))

        numerical = (
            torch.cat(blocks, dim=-1)
            if blocks
            else torch.zeros((*leading_shape, 0), dtype=dtype, device=device)
        )
        return table.__class__(
            columns={Stype.numerical: tuple(names)},
            numerical=numerical,
        )
