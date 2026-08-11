# ruff: noqa: D205

from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING, Any, NamedTuple, cast

import pyarrow.compute as pc
import torch
from torch import Tensor

from sdm import StringTensor, Stype, TableTensor
from sdm.processing import Processor

if TYPE_CHECKING:
    import sentence_transformers
    from cudf.core.character_normalizer import CharacterNormalizer
    from cudf.core.wordpiece_tokenize import WordPieceVocabulary


class _WordPieceTokenizer(NamedTuple):
    wpt: WordPieceVocabulary
    normalizer: CharacterNormalizer
    cls_id: int
    sep_id: int
    pad_id: int
    max_length: int


def _build_wp_tokenizer(
    model: sentence_transformers.SentenceTransformer,
) -> _WordPieceTokenizer | None:
    if not torch.cuda.is_available():
        return None
    if importlib.util.find_spec("cudf") is None:
        return None

    from transformers import PreTrainedTokenizerFast  # noqa: PLC0415

    tokenizer = model.tokenizer
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        return None
    if tokenizer.backend_tokenizer.model.__class__.__name__ != "WordPiece":
        return None

    import cudf
    from cudf.core.character_normalizer import (
        CharacterNormalizer,
    )
    from cudf.core.wordpiece_tokenize import (
        WordPieceVocabulary,
    )

    vocab_tokens = tokenizer.convert_ids_to_tokens(range(tokenizer.vocab_size))
    return _WordPieceTokenizer(
        wpt=WordPieceVocabulary(cudf.Series(vocab_tokens)),
        normalizer=CharacterNormalizer(
            do_lower=tokenizer.do_lower_case,
            special_tokens=cudf.Series(tokenizer.all_special_tokens),
        ),
        cls_id=tokenizer.cls_token_id,
        sep_id=tokenizer.sep_token_id,
        pad_id=tokenizer.pad_token_id,
        max_length=tokenizer.model_max_length,
    )


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
        batch_size: Batch size for the forward pass. Adjusting the batch size
            can significantly improve processing speed. The optimal value
            depends on your hardware, model size, precision, and input length.
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
        self._word_piece_tokenizer = _build_wp_tokenizer(model)

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
            text = cast(StringTensor, table.text.movedim(-1, 0).reshape(-1))
            if self._word_piece_tokenizer is not None:
                emb = self._encode_gpu(text)
            else:
                array = text.to_arrow()
                if text.is_nullable:
                    array = pc.fill_null(array, "")
                emb = self._model.module.encode(
                    array.to_pylist(),
                    show_progress_bar=False,
                    convert_to_tensor=True,
                    device=str(table.device),
                    batch_size=self.batch_size,
                )
                assert isinstance(emb, Tensor)
            emb = emb.to(device=table.device, dtype=table.dtype)
            numerical = (
                emb.reshape(len(columns), *batch_shape, self._embedding_dim)
                .movedim(0, -2)
                .reshape(*batch_shape, len(output_columns))
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

    def _encode_gpu(self, text: StringTensor) -> Tensor:
        tokenizer = self._word_piece_tokenizer
        assert tokenizer is not None
        device = text.device
        num_strings = text.numel()

        _t: list[tuple[str, float]] = []

        def _sync_ms() -> float:
            torch.cuda.synchronize()
            import time  # noqa: PLC0415

            return time.perf_counter()

        t0 = _sync_ms()

        text_series = text.to_cudf()
        if text.is_nullable:
            text_series = text_series.fillna("")

        t1 = _sync_ms()
        _t.append(("to_cudf + fillna", t1 - t0))

        normalized = tokenizer.normalizer.normalize(text_series)
        token_lists = tokenizer.wpt.tokenize(normalized)

        t2 = _sync_ms()
        _t.append(("normalize + tokenize", t2 - t1))

        flat_values = torch.from_dlpack(token_lists.list.leaves.to_cupy())
        raw_lengths = torch.from_dlpack(token_lists.list.len().to_cupy()).to(
            torch.long
        )

        t3 = _sync_ms()
        _t.append(("extract flat values + lengths", t3 - t2))

        # Source offsets into the flat token buffer
        offsets = torch.zeros(num_strings + 1, device=device, dtype=torch.long)
        torch.cumsum(raw_lengths, dim=0, out=offsets[1:])

        lengths = raw_lengths.clamp(max=tokenizer.max_length - 2)
        seq_len = int(lengths.max()) + 2  # [CLS] + tokens + [SEP]

        input_ids = torch.full(
            (num_strings, seq_len),
            tokenizer.pad_id,
            device=device,
            dtype=torch.long,
        )
        input_ids[:, 0] = tokenizer.cls_id

        # Scatter truncated token IDs into positions 1..lengths[i]+1
        row_idx = torch.arange(num_strings, device=device).repeat_interleave(
            lengths
        )
        total = row_idx.shape[0]
        starts = lengths.cumsum(0) - lengths
        within_row = torch.arange(
            total, device=device
        ) - starts.repeat_interleave(lengths)
        src_idx = within_row + offsets[:-1].repeat_interleave(lengths)
        input_ids[row_idx, within_row + 1] = flat_values[src_idx].to(
            torch.long
        )

        input_ids[torch.arange(num_strings, device=device), lengths + 1] = (
            tokenizer.sep_id
        )
        attention_mask = (input_ids != tokenizer.pad_id).to(torch.long)

        t4 = _sync_ms()
        _t.append(("scatter + pad", t4 - t3))

        embeddings = torch.empty(
            num_strings,
            self._embedding_dim,
            device=device,
            dtype=torch.float,
        )
        with torch.inference_mode():
            for batch_start in range(0, num_strings, self.batch_size):
                batch_end = min(batch_start + self.batch_size, num_strings)
                batch_seq_len = int(lengths[batch_start:batch_end].max()) + 2
                features: dict[str, Tensor] = {
                    "input_ids": input_ids[
                        batch_start:batch_end, :batch_seq_len
                    ],
                    "attention_mask": attention_mask[
                        batch_start:batch_end, :batch_seq_len
                    ],
                }
                for module in self._model.module:
                    features = module(features)
                embeddings[batch_start:batch_end] = features[
                    "sentence_embedding"
                ]

        t5 = _sync_ms()
        _t.append(("forward pass", t5 - t4))

        total_s = sum(s for _, s in _t)
        print(f"\n[_encode_gpu] num_strings={num_strings}")  # noqa: T201
        for label, s in _t:
            print(  # noqa: T201
                f"  {label:30s} {s:8.4f}s  {s / total_s * 100:5.1f}%"
            )
        print(f"  {'total':30s} {total_s:8.4f}s")  # noqa: T201
        return embeddings
