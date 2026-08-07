from __future__ import annotations

from typing import Any, cast

import torch
from torch import Tensor

from sdm.processing.base import Processor
from sdm.stype import Stype
from sdm.tensor import StringTensor, TableTensor


class _ModuleReference(torch.nn.Module):
    def __init__(self, model_name: str) -> None:
        super().__init__()
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415

        self.model: Any = SentenceTransformer(model_name)

    def __deepcopy__(self, memo: dict[int, Any]) -> _ModuleReference:
        return self


class EmbedText(Processor):
    r"""Embed text columns with a Sentence Transformers model.

    Args:
        model_name: Model name or local path passed to
            :class:`sentence_transformers.SentenceTransformer`.
        batch_size: Number of text cells encoded in each model batch. If
            ``None``, use the model default.
    """

    requires_fit = False
    supported_stypes = frozenset({Stype.text})

    def __init__(
        self,
        model_name: str,
        *,
        batch_size: int | None = None,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = _ModuleReference(model_name)
        embedding_dim = self._model.model.get_embedding_dimension()
        assert isinstance(embedding_dim, int)
        self._embedding_dim = embedding_dim

    def _transform(self, table: TableTensor) -> TableTensor:
        columns = table.columns[Stype.text]
        batch_shape = table.text.shape[:-1]
        output_columns = tuple(
            f"{column}_{index}"
            for column in columns
            for index in range(self._embedding_dim)
        )

        if table.text.numel() == 0:
            numerical = torch.empty(
                (*batch_shape, len(output_columns)),
                dtype=torch.get_default_dtype(),
                device=table.device,
            )
        else:
            text = cast(
                StringTensor,
                table.text.movedim(-1, 0).reshape(-1),
            )
            strings = [value or "" for value in text.tolist()]
            encode_kwargs = {}
            if self.batch_size is not None:
                encode_kwargs["batch_size"] = self.batch_size
            embeddings = cast(
                Tensor,
                self._model.model.encode(
                    strings,
                    show_progress_bar=False,
                    convert_to_tensor=True,
                    device=str(table.device),
                    **encode_kwargs,
                ),
            ).to(dtype=torch.get_default_dtype(), device=table.device)

            # [num_columns, *batch_shape, embedding_dim]
            numerical = (
                embeddings.reshape(
                    len(columns),
                    *batch_shape,
                    self._embedding_dim,
                )
                .movedim(0, -2)
                .reshape(*batch_shape, len(output_columns))
            )

        return TableTensor(
            columns={Stype.numerical: output_columns},
            numerical=numerical,
        )
