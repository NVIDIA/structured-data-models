import math
import re
from itertools import accumulate
from typing import cast

import pyarrow as pa
import pyarrow.compute as pc
import torch
from torch import Tensor
from torch.utils.dlpack import from_dlpack

from sdm import StringTensor, Stype, TableTensor
from sdm.processing import EnsembleProcessor
from sdm.tensor import EnsembleTable
from sdm.tensor.io import arrow_as_tensor


class _TFIDFState(torch.nn.Module):
    def __init__(
        self,
        vocabularies: list[pa.Array],
        idfs: list[Tensor],
    ) -> None:
        super().__init__()
        self.vocabularies = vocabularies
        for column, idf in enumerate(idfs):
            self.register_buffer(f"idf_{column}", idf)


class TFIDF(EnsembleProcessor):
    """Encode text columns as character n-gram TF-IDF vectors.

    Tokenization follows scikit-learn's ``char_wb`` analyzer: whitespace-
    delimited words are space-padded before windowing, so a word shorter than
    ``n`` still yields one n-gram. Fit learns a vocabulary and smoothed idf
    weights per text column. Transform replaces text with concatenated
    numerical features (one per retained n-gram), applies those idf weights,
    L2-normalizes each row, and ignores n-grams unseen at fit time.
    When fitted on an :class:`~sdm.tensor.EnsembleTable`, distinct member
    tables learn independent vocabularies and can produce different numerical
    schemas; members assigned the same table share fitted state.

    Args:
        ngram_range: Inclusive ``(min_n, max_n)`` character-window sizes.
        max_features: If set, keep only this many most frequent n-grams per
            column. ``None`` keeps the full vocabulary.
        lowercase: If ``True``, lowercase text before tokenizing.
    """

    supported_stypes = frozenset({Stype.text})

    def __init__(
        self,
        *,
        ngram_range: tuple[int, int],
        max_features: int | None = None,
        lowercase: bool = True,
    ) -> None:
        super().__init__()
        min_n, max_n = ngram_range
        if min_n < 1 or max_n < min_n:
            raise ValueError("ngram_range must satisfy 1 <= min_n <= max_n.")
        if max_features is not None and max_features <= 0:
            raise ValueError("max_features must be positive or None.")
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.lowercase = lowercase
        self._states = torch.nn.ModuleList()
        self._member_state_ids: tuple[int, ...] = ()

    def _character_ngrams(
        self,
        tensor: StringTensor,
        ngram_range: tuple[int, int],
        *,
        lowercase: bool = True,
    ) -> tuple[StringTensor, Tensor]:
        if tensor.dim() != 1:
            raise NotImplementedError(
                "Expected tensor to be one-dimensional "
                f"(got {tensor.dim()}D tensor)"
            )

        if tensor.is_cuda:
            return self._character_ngrams_cuda(tensor, ngram_range, lowercase)

        min_n, max_n = ngram_range
        whitespace = re.compile(r"\s\s+")
        ngrams: list[str] = []
        offsets: list[int] = [0]
        for document in tensor.to_arrow().to_pylist():
            if document is None:
                offsets.append(len(ngrams))
                continue
            if lowercase:
                document = document.lower()
            document = whitespace.sub(" ", document)
            for word in document.split():
                word = " " + word + " "
                word_len = len(word)
                for n in range(min_n, max_n + 1):
                    offset = 0
                    ngrams.append(word[offset : offset + n])
                    while offset + n < word_len:
                        offset += 1
                        ngrams.append(word[offset : offset + n])
                    if offset == 0:  # word shorter than n: count it once
                        break
            offsets.append(len(ngrams))

        flat = tensor.from_arrow(
            pa.array(ngrams, type=pa.large_string()),
            device=tensor.device,
        )
        offset = torch.tensor(offsets, dtype=torch.int64, device=tensor.device)
        return flat, offset

    def _character_ngrams_cuda(
        self,
        tensor: StringTensor,
        ngram_range: tuple[int, int],
        lowercase: bool = True,
    ) -> tuple[StringTensor, Tensor]:
        import cudf
        import cupy as cp

        min_n, max_n = ngram_range
        n_docs = tensor.numel()
        s = tensor.to_cudf()  # n_docs documents
        if tensor.is_nullable:
            s = s.fillna("")
        if lowercase:
            s = s.str.lower()
        # Collapse every whitespace run to a single space and trim, so each
        # document splits into words on single spaces (matches str.split()).
        s = s.str.replace(r"\s+", " ", regex=True).str.strip()

        words = s.str.split(" ")
        doc_index = cudf.Series(
            cp.repeat(cp.arange(n_docs), words.list.len().to_cupy())
        )
        flat_words = words.explode().reset_index(drop=True)
        keep = flat_words.str.len() > 0
        flat_words = flat_words[keep].reset_index(drop=True)
        doc_index = doc_index[keep].reset_index(drop=True)
        padded = " " + flat_words + " "
        pad_len = padded.str.len()

        parts: list[cudf.DataFrame] = []
        # 'character_ngrams' raises when no word is long enough for 'n'.
        for n in range(min_n, max_n + 1):
            eligible = pad_len >= n
            padded_eligable = padded[eligible]
            if len(padded_eligable) == 0:
                continue
            grams = padded_eligable.str.character_ngrams(n, as_list=True)
            long = cudf.DataFrame(
                {
                    "doc": doc_index[eligible],
                    "gram": grams,
                }
            ).explode("gram")
            parts.append(long.dropna(subset=["gram"]))

        # word shorter than min_n: count it once
        short = pad_len < min_n
        parts.append(
            cudf.DataFrame({"doc": doc_index[short], "gram": padded[short]})
        )

        flat = cudf.concat(parts, ignore_index=True).sort_values("doc")

        counts = (
            flat.groupby("doc").size().reindex(range(n_docs), fill_value=0)
        )
        offset = torch.zeros(
            n_docs + 1,
            dtype=torch.int64,
            device=tensor.device,
        )
        offset[1:] = from_dlpack(counts.cumsum().astype("int64").to_dlpack())

        ngrams = tensor.from_cudf(
            flat["gram"].reset_index(drop=True), device=tensor.device
        )
        return ngrams, offset

    def _learn_state(
        self,
        table: TableTensor,
    ) -> _TFIDFState:
        device = table.text.device
        vocabularies: list[pa.Array] = []
        idfs: list[Tensor] = []

        for column in range(table.text.size(-1)):
            column_text = cast(
                StringTensor,
                table.text[..., column].reshape(-1),
            )
            flat, offsets = self._character_ngrams(
                column_text,
                self.ngram_range,
                lowercase=self.lowercase,
            )
            n_docs = offsets.numel() - 1

            # Factorize n-grams into codes + the unique vocabulary
            if flat.is_cuda:
                import cudf

                values = flat.to_cudf()
                vocabulary = values.drop_duplicates(
                    ignore_index=True
                ).to_arrow()
                encoded = values.astype(
                    cudf.CategoricalDtype(categories=vocabulary)
                )
                codes = torch.from_dlpack(
                    encoded.cat.codes.astype("int64").to_dlpack()
                ).to(device)
            else:
                encoded = flat.to_arrow().dictionary_encode()
                vocabulary = encoded.dictionary
                codes = arrow_as_tensor(
                    encoded.indices, dtype=torch.int64, device=device
                )  # [n_ngrams]
            vocab_size = len(vocabulary)
            # Map each n-gram back to its document via the offsets.
            doc_ids = torch.repeat_interleave(
                torch.arange(n_docs, device=device),
                offsets.diff(),
            )  # [n_ngrams]

            # Smoothed idf per n-gram from its document frequency: count each
            # n-gram once per document, then apply sklearn's smoothing.
            if vocab_size == 0:
                idf = torch.empty(0, device=device)
            else:
                unique_codes = (
                    doc_ids * vocab_size + codes
                ).unique() % vocab_size
                document_freq = torch.bincount(
                    unique_codes, minlength=vocab_size
                )
                idf = ((1 + n_docs) / (1 + document_freq)).log() + 1.0
            term_counts = torch.bincount(
                codes, minlength=vocab_size
            )  # [vocab_size]

            # Keep the max_features n-grams with the highest term frequency,
            # matching scikit-learn: rank by corpus occurrence, break ties by
            # vocabulary order via a stable sort.
            if (
                self.max_features is not None
                and len(vocabulary) > self.max_features
            ):
                ranked = term_counts.argsort(descending=True, stable=True)
                keep = ranked[: self.max_features].sort().values
                vocabulary = vocabulary.take(pa.array(keep.tolist()))
                idf = idf[keep]
            vocabularies.append(vocabulary)
            idfs.append(idf)

        return _TFIDFState(vocabularies, idfs)

    def _fit_ensemble(
        self,
        ensemble_table: EnsembleTable,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        states = torch.nn.ModuleList()
        member_state_ids: list[int] = []
        state_ids: dict[tuple[int, int], int] = {}

        # TODO: Replace direct `_locations` access with a public
        # `EnsembleTable` iterator over stored tables and their logical member
        # IDs, then use the same abstraction when transforming.
        for member_id in range(ensemble_table.num_members):
            location = ensemble_table._locations[member_id]
            state_id = state_ids.get(location)
            if state_id is None:
                state_id = len(states)
                state_ids[location] = state_id
                states.append(
                    self._learn_state(ensemble_table.table(member_id))
                )
            member_state_ids.append(state_id)

        self._states = states
        self._member_state_ids = tuple(member_state_ids)

    def _transform_ensemble(
        self,
        ensemble_table: EnsembleTable,
    ) -> EnsembleTable:
        if len(self._member_state_ids) != ensemble_table.num_members:
            raise RuntimeError(
                "TFIDF must be fitted with the same number of "
                "ensemble members before transform."
            )

        output_tables: list[TableTensor] = []
        member_table_ids: list[int] = []
        transformed: dict[tuple[tuple[int, int], int], int] = {}
        # TODO: Benchmark batching compatible query tables that share fitted
        # state on CUDA with cuDF. It was about 10% slower for four 50,000-row
        # tables on CPU.
        for member_id, state_id in enumerate(self._member_state_ids):
            key = (ensemble_table._locations[member_id], state_id)
            table_id = transformed.get(key)
            if table_id is None:
                state = cast(_TFIDFState, self._states[state_id])
                table_id = len(output_tables)
                transformed[key] = table_id
                output_tables.append(
                    self._encode(
                        ensemble_table.table(member_id),
                        state,
                    )
                )
            member_table_ids.append(table_id)

        return EnsembleTable.from_tables(
            output_tables,
            member_table_ids,
        )

    def _encode(
        self,
        table: TableTensor,
        state: _TFIDFState,
    ) -> TableTensor:
        device = table.text.device
        dtype = (
            state.get_buffer("idf_0").dtype
            if state.vocabularies
            else torch.get_default_dtype()
        )
        text_names = table.columns[Stype.text]
        leading_shape = table.text.shape[:-1]
        n_rows = math.prod(leading_shape)

        vocab_sizes = [len(vocabulary) for vocabulary in state.vocabularies]
        column_offsets = [0, *accumulate(vocab_sizes)]
        total_width = column_offsets[-1]
        numerical = torch.zeros(
            (*leading_shape, total_width),
            dtype=dtype,
            device=device,
        )
        flat_numerical = numerical.view(n_rows, total_width)
        names: list[str] = []
        for column in range(table.text.size(-1)):
            vocabulary = state.vocabularies[column]
            idf = getattr(state, f"idf_{column}")
            vocab_size = vocab_sizes[column]
            column_start = column_offsets[column]
            column_slice = flat_numerical[
                :, column_start : column_offsets[column + 1]
            ]

            if vocab_size > 0:
                column_text = cast(
                    StringTensor,
                    table.text[..., column].reshape(-1),
                )
                flat, offsets = self._character_ngrams(
                    column_text,
                    self.ngram_range,
                    lowercase=self.lowercase,
                )
                # Map n-grams to fitted vocab indices; unseen -> -1 (dropped).
                if flat.is_cuda:
                    import cudf

                    encoded = (
                        flat.to_cudf()
                        .astype(cudf.CategoricalDtype(categories=vocabulary))
                        .cat.codes
                    )
                    codes = torch.from_dlpack(
                        encoded.astype("int64").to_cupy(na_value=-1)
                    ).to(device)
                else:
                    codes = arrow_as_tensor(
                        pc.call_function(
                            "index_in",
                            [flat.to_arrow()],
                            options=pc.SetLookupOptions(value_set=vocabulary),
                        ).fill_null(-1),
                        dtype=torch.int64,
                        device=device,
                    )  # [n_ngrams]
                doc_ids = torch.repeat_interleave(
                    torch.arange(n_rows, device=device),
                    offsets.diff(),
                )  # [n_ngrams]
                mask = (codes >= 0) & (codes < vocab_size)
                flat_index = doc_ids[mask] * vocab_size + codes[mask]
                counts = column_slice.new_zeros(n_rows, vocab_size)
                counts.view(-1).scatter_add_(
                    0,
                    flat_index,
                    torch.ones_like(flat_index, dtype=dtype),
                )
                counts.mul_(idf)
                norm = counts.norm(dim=1, keepdim=True).clamp_min_(1e-12)
                column_slice.copy_(counts.div_(norm))
            names.extend(
                f"{text_names[column]}_{i}" for i in range(vocab_size)
            )

        return table.__class__(
            columns={Stype.numerical: tuple(names)},
            numerical=numerical,
        )

    def __repr__(self, *, indent: int = 0) -> str:
        return (
            f"{' ' * indent}{self.__class__.__name__}("
            f"ngram_range={self.ngram_range}, "
            f"max_features={self.max_features}, "
            f"lowercase={self.lowercase}"
            ")"
        )
