from __future__ import annotations

from typing import Any, cast

import torch
from torch import Tensor

from sdm.processing.base import Processor
from sdm.stype import Stype
from sdm.tensor import StringTensor, TableTensor


class _ModuleReference(torch.nn.Module):
    def __init__(self, module: torch.nn.Module) -> None:
        super().__init__()
        self.module = module

    def __deepcopy__(self, memo: dict[int, Any]) -> _ModuleReference:
        return type(self)(self.module)


class EmbedText(Processor):
    r"""Embed text columns with a Sentence Transformers model.

    Args:
        model_name: Model name or local path passed to
            :class:`sentence_transformers.SentenceTransformer`.
        batch_size: Batch size passed to `SentenceTransformer.encode
            <https://sbert.net/docs/package_reference/sentence_transformer/model.html#sentence_transformers.sentence_transformer.model.SentenceTransformer.encode>`_.
            If ``None``, use the model default.
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
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415

        self.batch_size = batch_size
        model: Any = SentenceTransformer(model_name)
        self._model = _ModuleReference(model)
        embedding_dim = model.get_embedding_dimension()
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
            model = cast(Any, self._model.module)
            embeddings = cast(
                Tensor,
                model.encode(
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
