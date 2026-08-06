from __future__ import annotations

from typing import Any, cast

import torch

from sdm.processing.ensemble import EnsembleProcessor
from sdm.stype import Stype
from sdm.tensor import EnsembleTable, StringTensor, TableTensor


class _ModuleReference(torch.nn.Module):
    """Preserve a module reference across deep copies."""

    def __init__(self, module: torch.nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.module(*args, **kwargs)

    def __deepcopy__(self, _memo: dict[int, Any]) -> _ModuleReference:
        return type(self)(self.module)


class EmbedText(EnsembleProcessor):
    r"""Embed text columns with a user-provided embedding model.

    Each text column is embedded cell-by-cell through ``embedding_model``.
    The model must return one embedding per text value as a
    :class:`torch.Tensor` with shape ``[n, embedding_dim]``. Embedding
    Embeddings are concatenated in column order into the numerical output.

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
        self._embedding_model = _ModuleReference(embedding_model)
        self._embedding_dim: int = embedding_dim

    def _transform(self, table: TableTensor) -> TableTensor:
        device = table.device
        dtype = torch.get_default_dtype()
        col_names = table.columns[Stype.text]
        batch_shape = table.text.shape[:-1]

        out_col_names: list[str] = []
        for col_name in col_names:
            out_col_names.extend(
                f"{col_name}_{i}" for i in range(self._embedding_dim)
            )

        numerical = torch.empty(
            (*batch_shape, len(out_col_names)),
            dtype=dtype,
            device=device,
        )
        if numerical.numel() != 0:
            # FIXME: There're currently multiple issues:
            # 1. Even though the same model gets applied to all columns, we run
            #    it once per column.
            # 2. The embedding_model currently must take in a dataframe and not
            #    a Tensor.
            for col_idx in range(len(col_names)):
                col_tensor = cast(
                    StringTensor,
                    table.text[..., col_idx].reshape(-1),
                )
                strings = (
                    col_tensor.to_cudf()
                    if col_tensor.is_cuda
                    else col_tensor.to_arrow()
                )
                block = self._embedding_model(strings)
                embeddings = block.to(device=device, dtype=dtype).reshape(
                    *batch_shape,
                    self._embedding_dim,
                )
                start_idx = col_idx * self._embedding_dim
                end_idx = start_idx + self._embedding_dim
                numerical[..., start_idx:end_idx] = embeddings

        return TableTensor(
            columns={Stype.numerical: tuple(out_col_names)},
            numerical=numerical,
        )

    def _transform_ensemble(
        self,
        ensemble_table: EnsembleTable,
    ) -> EnsembleTable:
        groups = list(ensemble_table)
        if not groups:
            return ensemble_table

        ref = groups[0]
        device = ref.device
        dtype = torch.get_default_dtype()
        col_names = ref.columns[Stype.text]
        n_cols = len(col_names)

        group_rows = [g.text.shape[:-1] for g in groups]
        group_flat_rows = [
            g.text[..., 0].reshape(-1).numel() for g in groups
        ]
        total_rows = sum(group_flat_rows)

        out_col_names: list[str] = []
        for col_name in col_names:
            out_col_names.extend(
                f"{col_name}_{i}" for i in range(self._embedding_dim)
            )
        total_width = len(out_col_names)

        flat_numerical = torch.empty(
            total_rows,
            total_width,
            dtype=dtype,
            device=device,
        )

        if total_rows > 0:
            for col_idx in range(n_cols):
                col_parts = [
                    cast(
                        StringTensor,
                        g.text[..., col_idx].reshape(-1),
                    )
                    for g in groups
                ]
                merged = cast(
                    StringTensor,
                    torch.cat(col_parts, dim=0),
                )
                strings = (
                    merged.to_cudf()
                    if merged.is_cuda
                    else merged.to_arrow()
                )
                block = self._embedding_model(strings)
                embeddings = block.to(
                    device=device, dtype=dtype,
                ).reshape(total_rows, self._embedding_dim)
                start = col_idx * self._embedding_dim
                end = start + self._embedding_dim
                flat_numerical[:, start:end] = embeddings

        outputs: list[TableTensor] = []
        offset = 0
        for batch_shape, n in zip(group_rows, group_flat_rows):
            numerical = flat_numerical[offset:offset + n].reshape(
                *batch_shape, total_width,
            )
            outputs.append(TableTensor(
                columns={Stype.numerical: tuple(out_col_names)},
                numerical=numerical,
            ))
            offset += n

        return ensemble_table.replace_groups(outputs)
