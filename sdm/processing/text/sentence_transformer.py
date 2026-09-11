# ruff: noqa: D205

from __future__ import annotations

import importlib.util
from functools import cached_property
from typing import TYPE_CHECKING, Any, cast

import pyarrow.compute as pc
import torch
from torch import Tensor

from sdm import StringTensor, Stype, TableTensor
from sdm._warnings import warn_once
from sdm.processing import Processor

if TYPE_CHECKING:
    import sentence_transformers
    from cudf.core.character_normalizer import CharacterNormalizer
    from cudf.core.wordpiece_tokenize import WordPieceVocabulary


# Hugging Face treats words with more than 100 Unicode characters as [UNK],
# while cuDF does so for words with 200 or more UTF-8 bytes.
_CUDF_WORDPIECE_MAX_BYTES = 200


class _CuDFTokenizer:
    def __init__(
        self,
        vocabulary: WordPieceVocabulary,
        normalizer: CharacterNormalizer,
        cls_token_id: int,
        sep_token_id: int,
        pad_token_id: int,
        max_length: int,
        max_input_chars_per_word: int,
    ) -> None:
        self._vocabulary = vocabulary
        self._normalizer = normalizer
        self._cls_token_id = cls_token_id
        self._sep_token_id = sep_token_id
        self._pad_token_id = pad_token_id
        self._max_length = max_length
        self._max_input_chars_per_word = max_input_chars_per_word

    @staticmethod
    def _is_supported(
        model: sentence_transformers.SentenceTransformer,
    ) -> bool:
        import tokenizers
        from sentence_transformers.sentence_transformer.modules import (
            Transformer,
        )
        from transformers import PreTrainedTokenizerFast

        module = model[0]
        tokenizer = model.tokenizer

        # Fall back to CPU tokenization unless this is a standard BERT
        # Transformer module with a fast tokenizer.
        if (
            type(module) is not Transformer
            or module.config.model_type != "bert"
            or not isinstance(tokenizer, PreTrainedTokenizerFast)
        ):
            return False

        if (
            model.modalities != ["text"]
            or model.default_prompt_name is not None
            or model.truncate_dim is not None
            or module.transformer_task != "feature-extraction"
            or module.backend != "torch"
            or module.processing_kwargs
            or module.can_flatten_inputs
            or module.unpad_inputs
        ):
            return False

        backend = tokenizer.backend_tokenizer
        wordpiece = backend.model
        normalizer = backend.normalizer

        # cuDF cannot currently construct an equivalent tokenizer directly
        # from the Hugging Face tokenizer configuration. Keep these checks
        # until it can.
        # Require the standard BERT WordPiece pipeline.
        if (
            type(wordpiece) is not tokenizers.models.WordPiece
            or type(normalizer) is not tokenizers.normalizers.BertNormalizer
            or type(backend.pre_tokenizer)
            is not tokenizers.pre_tokenizers.BertPreTokenizer
            or type(backend.post_processor)
            is not tokenizers.processors.TemplateProcessing
            or (
                normalizer.clean_text,
                normalizer.handle_chinese_chars,
                normalizer.strip_accents,
            )
            != (True, True, None)
            or (wordpiece.unk_token, wordpiece.continuing_subword_prefix)
            != ("[UNK]", "##")
            or (tokenizer.padding_side, tokenizer.truncation_side)
            != ("right", "right")
            or tokenizer.model_max_length != module.max_seq_length
        ):
            return False

        vocab = backend.get_vocab(with_added_tokens=False)
        added_tokens = tuple(backend.get_added_tokens_decoder().values())
        special_tokens = (
            (tokenizer.cls_token, tokenizer.cls_token_id),
            (tokenizer.sep_token, tokenizer.sep_token_id),
            (tokenizer.pad_token, tokenizer.pad_token_id),
            (tokenizer.unk_token, tokenizer.unk_token_id),
        )

        # Vocabulary IDs and added-token behavior must match cuDF's lookup.
        if (
            backend.get_vocab_size(with_added_tokens=True) != len(vocab)
            or set(vocab.values()) != set(range(len(vocab)))
            or not all(
                vocab.get(token) == token_id
                for token, token_id in special_tokens
            )
            or {token.content for token in added_tokens}
            != set(tokenizer.all_special_tokens)
            or not all(
                token.special
                and not token.normalized
                and not token.lstrip
                and not token.rstrip
                and not token.single_word
                for token in added_tokens
            )
        ):
            return False

        encoded = backend.encode(tokenizer.unk_token, add_special_tokens=True)
        return (
            backend.encode(
                tokenizer.unk_token,
                add_special_tokens=False,
            ).ids
            == [tokenizer.unk_token_id]
            and encoded.ids
            == [
                tokenizer.cls_token_id,
                tokenizer.unk_token_id,
                tokenizer.sep_token_id,
            ]
            and not any(encoded.type_ids)
        )

    @classmethod
    def build(
        cls,
        model: sentence_transformers.SentenceTransformer,
    ) -> _CuDFTokenizer | None:
        if not cls._is_supported(model):
            return None

        tokenizer = model.tokenizer
        backend = tokenizer.backend_tokenizer
        wordpiece = backend.model
        normalizer = backend.normalizer
        vocab = backend.get_vocab(with_added_tokens=False)

        if importlib.util.find_spec("cudf") is None:
            warn_once(
                key="on-device-tokenization-available-but-cudf-unavailable",
                message=(
                    "Falling back to CPU-based tokenization because cuDF is "
                    "not installed. Install cuDF to enable faster CUDA-based "
                    "tokenization without device synchronization."
                ),
            )
            return None

        import cudf
        from cudf.core.character_normalizer import CharacterNormalizer
        from cudf.core.wordpiece_tokenize import WordPieceVocabulary

        vocab_tokens = [
            token for token, _ in sorted(vocab.items(), key=lambda x: x[1])
        ]
        return cls(
            vocabulary=WordPieceVocabulary(cudf.Series(vocab_tokens)),
            normalizer=CharacterNormalizer(
                do_lower=normalizer.lowercase,
                special_tokens=cudf.Series(tokenizer.all_special_tokens),
            ),
            cls_token_id=tokenizer.cls_token_id,
            sep_token_id=tokenizer.sep_token_id,
            pad_token_id=tokenizer.pad_token_id,
            max_length=tokenizer.model_max_length,
            max_input_chars_per_word=wordpiece.max_input_chars_per_word,
        )

    def tokenize(self, text: StringTensor) -> tuple[Tensor, Tensor, Tensor]:
        device = text.device
        ser = text.to_cudf()
        ser = ser.fillna("") if text.is_nullable else ser
        ser = self._normalizer.normalize(ser)
        ser = self._vocabulary.tokenize(ser)

        flat_values = torch.from_dlpack(ser.list.leaves.to_cupy())
        lengths = torch.from_dlpack(ser.list.len().to_cupy())

        num_strings = text.numel()
        offsets = torch.empty(
            num_strings + 1,
            device=device,
            dtype=torch.int32,
        )
        offsets[0] = 0
        torch.cumsum(lengths, dim=0, out=offsets[1:])
        lengths.clamp_(max=self._max_length - 2).add_(2)
        return flat_values, offsets, lengths

    def batch(
        self,
        flat_values: Tensor,
        offsets: Tensor,
        lengths: Tensor,
        indices: Tensor,
        max_length: int,
    ) -> dict[str, Tensor]:
        device = flat_values.device
        lengths = lengths[indices].sub_(2)
        input_ids = torch.full(
            (indices.numel(), max_length),
            self._pad_token_id,
            device=device,
            dtype=torch.int32,
        )
        input_ids[:, 0] = self._cls_token_id

        if flat_values.numel() > 0:
            max_content = max_length - 2
            col_idx = torch.arange(
                max_content,
                device=device,
                dtype=torch.int32,
            ).unsqueeze(0)
            padding = col_idx >= lengths.unsqueeze(1)
            src = col_idx + offsets[indices].unsqueeze(1)
            src.masked_fill_(padding, 0)
            tokens = input_ids[:, 1 : max_content + 1]
            torch.where(
                padding,
                tokens,
                flat_values[src],
                out=tokens,
            )

        input_ids[
            torch.arange(indices.numel(), device=device), lengths.add_(1)
        ] = self._sep_token_id
        attention_mask = torch.empty_like(input_ids)
        torch.lt(
            torch.arange(
                max_length, device=device, dtype=torch.int32
            ).unsqueeze(0),
            lengths.add_(1).unsqueeze(1),
            out=attention_mask,
        )

        return {"input_ids": input_ids, "attention_mask": attention_mask}


class _Encoder(torch.nn.Module):
    def __init__(
        self,
        model: sentence_transformers.SentenceTransformer,
        batch_size: int,
        embedding_dim: int,
    ) -> None:
        super().__init__()
        self._model = model
        self._batch_size = batch_size
        self._embedding_dim = embedding_dim

    @cached_property
    def _cudf_tokenizer(self) -> _CuDFTokenizer | None:
        return _CuDFTokenizer.build(self._model)

    def __deepcopy__(self, memo: dict[int, Any]) -> _Encoder:
        return self

    def _forward_cpu(
        self,
        text: StringTensor,
        out: Tensor | None = None,
    ) -> Tensor:
        array = text.to_arrow()
        if text.is_nullable:
            array = pc.fill_null(array, "")
        emb = self._model.encode(
            array.to_pylist(),
            show_progress_bar=False,
            convert_to_tensor=True,
            device=str(text.device),
            batch_size=self._batch_size,
        )
        assert isinstance(emb, Tensor)
        return emb if out is None else out.copy_(emb.reshape_as(out))

    @staticmethod
    def _write_embeddings(
        out: Tensor,
        indices: Tensor,
        embeddings: Tensor,
    ) -> None:
        if out.dim() == 2:
            out[indices] = embeddings.to(out.dtype)
        else:
            # A numerical prefix leaves gaps between rows of text embeddings.
            columns = out.size(1)
            rows = indices.div(columns, rounding_mode="floor")
            out[rows, indices.remainder(columns)] = embeddings.to(out.dtype)

    def _forward_cudf(
        self,
        text: StringTensor,
        out: Tensor | None = None,
        indices: Tensor | None = None,
    ) -> Tensor:
        assert self._cudf_tokenizer is not None
        self._model.eval()
        flat_values, offsets, seq_lengths = self._cudf_tokenizer.tokenize(text)
        sort_idx = seq_lengths.argsort()

        device = text.device
        num_strings = text.numel()
        batch_ends = torch.arange(
            self._batch_size,
            num_strings + self._batch_size,
            self._batch_size,
            device=device,
        )
        batch_ends.clamp_(max=num_strings)
        batch_ends -= 1
        # Triggers a host device sync to get chunk sizes to
        # minimize the padding
        batch_max_lengths = seq_lengths[sort_idx[batch_ends]].tolist()

        if out is None:
            out = torch.empty(
                num_strings,
                self._embedding_dim,
                device=device,
                dtype=torch.float32,
            )
        # Amortize token gathering while bounding padding memory.
        batches_per_window = 16
        for first in range(0, len(batch_max_lengths), batches_per_window):
            last = min(first + batches_per_window, len(batch_max_lengths))
            window_indices = sort_idx[
                first * self._batch_size : last * self._batch_size
            ]
            tokens = self._cudf_tokenizer.batch(
                flat_values=flat_values,
                offsets=offsets,
                lengths=seq_lengths,
                indices=window_indices,
                max_length=batch_max_lengths[last - 1],
            )
            for i in range(first, last):
                start = (i - first) * self._batch_size
                end = start + self._batch_size
                batch_indices = window_indices[start:end]
                features = {
                    name: tensor[start:end, : batch_max_lengths[i]]
                    for name, tensor in tokens.items()
                }
                features = self._model(features)
                self._write_embeddings(
                    out=out,
                    indices=(
                        batch_indices
                        if indices is None
                        else indices[batch_indices]
                    ),
                    embeddings=features["sentence_embedding"].float(),
                )
                del features
            del tokens

        return out

    @torch.no_grad()
    def forward(self, text: StringTensor, out: Tensor | None = None) -> Tensor:
        if text.device.type == "cpu":
            return self._forward_cpu(text, out=out)

        tokenizer = self._cudf_tokenizer
        if tokenizer is None:
            return self._forward_cpu(text, out=out)

        series = text.to_cudf()
        series = series.fillna("") if text.is_nullable else series
        words = tokenizer._normalizer.normalize(series).str.tokenize()

        # CPU rejects long words by Unicode characters, while cuDF uses UTF-8
        # bytes. The following lines find rows where those decisions differ.
        # Remove this fallback if cuDF exposes the tokenizer's
        # max_input_chars_per_word setting.
        use_cpu = (words.str.len() > tokenizer._max_input_chars_per_word) != (
            words.str.byte_count() >= _CUDF_WORDPIECE_MAX_BYTES
        )
        cpu_indices = torch.from_dlpack(words[use_cpu].index.unique().values)
        del series, words, use_cpu

        num_strings = text.numel()
        if cpu_indices.numel() == 0:
            return self._forward_cudf(text, out=out)
        if cpu_indices.numel() == num_strings:
            return self._forward_cpu(text, out=out)

        use_cudf = torch.ones(
            num_strings,
            device=text.device,
            dtype=torch.bool,
        )
        use_cudf[cpu_indices] = False
        cudf_indices = use_cudf.nonzero().flatten()
        del use_cudf

        # Tokenize each subset with the matching path, then restore the
        # original input order through indexed assignment.
        if out is None:
            out = torch.empty(
                num_strings,
                self._embedding_dim,
                device=text.device,
                dtype=torch.float32,
            )
        cudf_text = cast(StringTensor, text[cudf_indices])
        self._forward_cudf(cudf_text, out=out, indices=cudf_indices)
        del cudf_text
        cpu_text = cast(StringTensor, text[cpu_indices])
        self._write_embeddings(
            out=out,
            indices=cpu_indices,
            embeddings=self._forward_cpu(cpu_text).float(),
        )
        return out


class SentenceTransformer(Processor):
    r"""Transform text columns with a
    :class:`sentence_transformers.SentenceTransformer` model.

    Args:
        model_name: Model name or local path passed to
            :class:`sentence_transformers.SentenceTransformer`
        batch_size: Batch size passed to
            :meth:`sentence_transformers.SentenceTransformer.encode`.
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
        import sentence_transformers

        self.batch_size = batch_size
        model = sentence_transformers.SentenceTransformer(model_name)
        embedding_dim = model.get_embedding_dimension()
        assert isinstance(embedding_dim, int)
        self._embedding_dim = embedding_dim
        self._model = _Encoder(model, batch_size, embedding_dim)

    def _transform(self, table: TableTensor) -> TableTensor:
        columns = table.columns[Stype.text]
        batch_shape = table.text.shape[:-1]
        output_columns = tuple(
            f"{column}__emb{i}"
            for column in columns
            for i in range(self._embedding_dim)
        )

        num_numerical = table.numerical.size(-1)
        num_output = num_numerical + len(output_columns)
        if (
            table.text.numel() == 0
            or num_numerical
            or (table.is_cuda and table.dtype != torch.float32)
        ):
            numerical = torch.empty(
                (*batch_shape, num_output),
                dtype=table.dtype,
                device=table.device,
            )
            numerical[..., :num_numerical].copy_(table.numerical)
            if table.text.numel():
                self._model(
                    cast(StringTensor, table.text.reshape(-1)),
                    out=numerical[..., num_numerical:].view(
                        -1, len(columns), self._embedding_dim
                    ),
                )
        else:
            text = cast(StringTensor, table.text.reshape(-1))
            emb = self._model(text)
            emb = emb.to(device=table.device, dtype=table.dtype)
            numerical = emb.reshape(*batch_shape, num_output)

        return TableTensor(
            columns={
                **table.columns,
                Stype.text: (),
                Stype.numerical: (
                    *table.columns[Stype.numerical],
                    *output_columns,
                ),
            },
            numerical=numerical,
            categorical=table.categorical,
            datetime=table.datetime,
            id=table.id,
        )
