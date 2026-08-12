# ruff: noqa: D205

from __future__ import annotations

import importlib.util
import json
from typing import TYPE_CHECKING, Any, NamedTuple, cast

import pyarrow.compute as pc
import torch
from torch import Tensor

from sdm import StringTensor, Stype, TableTensor
from sdm.processing import Processor

if TYPE_CHECKING:
    import cudf
    import sentence_transformers
    from cudf.core.byte_pair_encoding import BytePairEncoder


class _BPETokenizer(NamedTuple):
    encoder: BytePairEncoder
    vocab: cudf.DataFrame
    bos_id: int
    eos_id: int
    pad_id: int
    unk_id: int
    max_length: int


def _build_bpe_tokenizer(
    model: sentence_transformers.SentenceTransformer,
) -> _BPETokenizer | None:
    if not torch.cuda.is_available():
        return None
    if importlib.util.find_spec("cudf") is None:
        return None

    from transformers import PreTrainedTokenizerFast  # noqa: PLC0415

    tokenizer = model.tokenizer
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        return None
    if tokenizer.backend_tokenizer.model.__class__.__name__ != "BPE":
        return None

    import cudf
    from cudf.core.byte_pair_encoding import BytePairEncoder

    tok_json = json.loads(tokenizer.backend_tokenizer.to_str())
    merges = tok_json["model"]["merges"]
    encoder = BytePairEncoder(cudf.Series(merges))

    vocab_dict = tokenizer.get_vocab()
    vocab = cudf.DataFrame(
        {
            "token": list(vocab_dict.keys()),
            "id": list(vocab_dict.values()),
        }
    ).astype({"id": "int32"})

    return _BPETokenizer(
        encoder=encoder,
        vocab=vocab,
        bos_id=tokenizer.bos_token_id,
        eos_id=tokenizer.eos_token_id,
        pad_id=tokenizer.pad_token_id,
        unk_id=tokenizer.unk_token_id or 3,
        max_length=model.max_seq_length or tokenizer.model_max_length,
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
        self._bpe_tokenizer = _build_bpe_tokenizer(model)

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
            if self._bpe_tokenizer is not None:
                emb = self._encode_bpe(text)
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

    def _encode_bpe(self, text: StringTensor) -> Tensor:
        bpe = self._bpe_tokenizer
        assert bpe is not None
        device = text.device
        num_strings = text.numel()

        text_series = text.to_cudf()
        if text.is_nullable:
            text_series = text_series.fillna("")

        # Byte-level pre-tokenization: prepend Ġ to words after spaces
        text_series = text_series.str.replace(" ", " Ġ")

        # BPE encode on GPU
        encoded = bpe.encoder(text_series)

        # Split into subword tokens and map to vocab IDs
        token_lists = encoded.str.split(" ")
        raw_lengths = torch.from_dlpack(token_lists.list.len().to_cupy()).to(
            torch.long
        )

        # Vocab lookup via left merge (preserves order)
        flat_tokens = token_lists.explode().to_frame("token")
        merged = flat_tokens.merge(bpe.vocab, on="token", how="left")
        flat_ids = merged["id"].fillna(bpe.unk_id).astype("int32")
        flat_values = torch.from_dlpack(flat_ids.to_cupy())

        # Source offsets into the flat token buffer
        offsets = torch.zeros(num_strings + 1, device=device, dtype=torch.long)
        torch.cumsum(raw_lengths, dim=0, out=offsets[1:])

        lengths = raw_lengths.clamp(max=bpe.max_length - 2)
        seq_len = int(lengths.max()) + 2  # [BOS] + tokens + [EOS]

        input_ids = torch.full(
            (num_strings, seq_len),
            bpe.pad_id,
            device=device,
            dtype=torch.long,
        )
        input_ids[:, 0] = bpe.bos_id

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
            bpe.eos_id
        )
        attention_mask = (input_ids != bpe.pad_id).to(torch.long)

        # Sort by length so batches have similar-length sequences
        sort_idx = lengths.argsort()
        sorted_input_ids = input_ids[sort_idx]
        sorted_attention_mask = attention_mask[sort_idx]
        sorted_lengths = lengths[sort_idx]

        # Precompute per-batch max lengths in one sync
        batch_ends = (
            torch.arange(
                self.batch_size,
                num_strings + self.batch_size,
                self.batch_size,
                device=device,
            ).clamp(max=num_strings)
            - 1
        )
        batch_max_lengths = (sorted_lengths[batch_ends] + 2).tolist()

        embeddings = torch.empty(
            num_strings,
            self._embedding_dim,
            device=device,
            dtype=torch.float,
        )
        with torch.inference_mode():
            for i, batch_start in enumerate(
                range(0, num_strings, self.batch_size)
            ):
                batch_end = min(batch_start + self.batch_size, num_strings)
                batch_seq_len = batch_max_lengths[i]
                features: dict[str, Tensor] = {
                    "input_ids": sorted_input_ids[
                        batch_start:batch_end, :batch_seq_len
                    ],
                    "attention_mask": sorted_attention_mask[
                        batch_start:batch_end, :batch_seq_len
                    ],
                }
                for module in self._model.module:
                    features = module(features)
                embeddings[sort_idx[batch_start:batch_end]] = features[
                    "sentence_embedding"
                ]
        return embeddings
