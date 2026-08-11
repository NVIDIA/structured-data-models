# ruff: noqa: D205

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, cast

import pyarrow.compute as pc
import torch
from torch import Tensor

from sdm import StringTensor, Stype, TableTensor
from sdm.processing import Processor

if TYPE_CHECKING:
    import sentence_transformers


class _ModuleReference(torch.nn.Module):
    def __init__(
        self,
        module: sentence_transformers.SentenceTransformer,
    ) -> None:
        super().__init__()
        self.module = module

    def __deepcopy__(self, memo: dict[int, Any]) -> _ModuleReference:
        return type(self)(self.module)


class SentenceTransformer(Processor):
    r"""Transform text columns with a
    :class:`sentence_transformers.SentenceTransformer
    <sentence_transformers.sentence_transformer.model.SentenceTransformer>`
    model.

    Args:
        model_name: Model name or local path passed to
            :class:`sentence_transformers.SentenceTransformer
            <sentence_transformers.sentence_transformer.model.SentenceTransformer>`.
        batch_size: Batch size passed to
            :meth:`sentence_transformers.SentenceTransformer.encode()
            <sentence_transformers.sentence_transformer.model.SentenceTransformer.encode>`.
            Adjusting the batch size can significantly improve processing
            speed. The optimal value depends on your hardware, model size,
            precision, and input length.
    """

    requires_fit = False
    handles_stypes = frozenset({Stype.text})

    def __init__(
        self,
        model_name: str,
        *,
        batch_size: int = 32,
    ) -> None:
        super().__init__()
        import sentence_transformers  # noqa: PLC0415

        self.batch_size = batch_size
        model = sentence_transformers.SentenceTransformer(model_name)
        embedding_dim = model.get_embedding_dimension()
        assert isinstance(embedding_dim, int)
        self._embedding_dim = embedding_dim
        self._model = _ModuleReference(model)

    def _transform(self, table: TableTensor) -> TableTensor:
        columns = table.columns[Stype.text]
        batch_shape = table.text.shape[:-1]
        output_columns = tuple(
            f"{column}__emb{i}"
            for column in columns
            for i in range(self._embedding_dim)
        )

        if table.text.numel() == 0:
            numerical = torch.empty(
                (*batch_shape, len(output_columns)),
                dtype=table.dtype,
                device=table.device,
            )
        else:
            t0 = time.perf_counter()

            text = cast(StringTensor, table.text.movedim(-1, 0).reshape(-1))

            t_reshape = time.perf_counter() - t0
            t0 = time.perf_counter()

            array = text.to_arrow()

            t_to_arrow = time.perf_counter() - t0
            t0 = time.perf_counter()

            if text.is_nullable:
                array = pc.fill_null(array, "")

            pylist = array.to_pylist()

            t_to_pylist = time.perf_counter() - t0
            t0 = time.perf_counter()

            emb = self._model.module.encode(
                pylist,
                show_progress_bar=False,
                convert_to_tensor=True,
                device=str(table.device),
                batch_size=self.batch_size,
            )

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_encode = time.perf_counter() - t0
            t0 = time.perf_counter()

            assert isinstance(emb, Tensor)
            emb = emb.to(device=table.device, dtype=table.dtype)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_to_device = time.perf_counter() - t0
            t0 = time.perf_counter()

            numerical = (
                emb.reshape(len(columns), *batch_shape, self._embedding_dim)
                .movedim(0, -2)
                .reshape(*batch_shape, len(output_columns))
            )

            t_reshape_out = time.perf_counter() - t0
            print(  # noqa: T201
                f"\n[SentenceTransformer._transform] "
                f"n_strings={len(pylist)}\n"
                f"  reshape input:  {t_reshape:.4f}s\n"
                f"  to_arrow:       {t_to_arrow:.4f}s\n"
                f"  to_pylist:      {t_to_pylist:.4f}s\n"
                f"  encode:         {t_encode:.4f}s\n"
                f"  to device/dtype:{t_to_device:.4f}s\n"
                f"  reshape output: {t_reshape_out:.4f}s"
            )

        out = torch.cat(
            [
                table.drop_stypes(Stype.text),
                TableTensor(
                    columns={Stype.numerical: output_columns},
                    numerical=numerical,
                ),
            ],
            dim=-1,
        )
        return cast(TableTensor, out)
