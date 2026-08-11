# GPU WordPiece Tokenization for SentenceTransformer

## Problem

The `SentenceTransformer` processor in `sdm/processing/text/sentence_transformer.py` converts text columns into embeddings. Today the entire pipeline — tokenization, forward pass, pooling — is delegated to `sentence_transformers.SentenceTransformer.encode()`. That method tokenizes on CPU (via HuggingFace's tokenizer), transfers token IDs to GPU, then runs the transformer forward pass on GPU.

Tokenization accounts for ~36% of wall time. The CPU tokenization also forces two unnecessary data movements: GPU strings must come down to CPU (`StringTensor` → Arrow → Python list), and the resulting token IDs must go back up to GPU.

## Goal

For models that use WordPiece tokenization (BERT-family), replace the CPU tokenization with cuDF's GPU-native `WordPieceVocabulary` tokenizer. The entire pipeline — from `StringTensor` input to embedding output — stays on GPU with no CPU detours.

BPE-based models (RoBERTa, GPT-2 family) and environments without cuDF fall back to the current `model.encode()` path. The user-facing API does not change.

## Background

### Current data flow (CPU tokenization)

```
StringTensor (GPU)
  → .to_arrow()            → CPU, Arrow array
  → .to_pylist()            → CPU, Python list[str]
  → model.encode(pylist)    → CPU tokenization, then GPU forward pass
  → embedding tensor (GPU)
```

### Proposed data flow (GPU tokenization)

```
StringTensor (GPU)
  → .to_cudf()                          zero-copy, stays on GPU
  → cudf CharacterNormalizer.normalize() GPU string normalization
  → cudf WordPieceVocabulary.tokenize()  GPU tokenization → Series of list[int32]
  → extract flat values + offsets        zero-copy via DLPack to torch
  → torch scatter                        truncate + [CLS]/[SEP] + pad in one pass
  → transformer + pooling forward pass   using model's own nn.Module weights
  → embedding tensor (GPU)
```

### Key libraries

- **cuDF** (`cudf`): GPU DataFrame library from RAPIDS. Already a runtime dependency of sdm. Provides `WordPieceVocabulary` and `CharacterNormalizer` for GPU-native WordPiece tokenization.
- **pylibcudf** (`pylibcudf`): Low-level C++/Cython bindings under cuDF. Already used in `StringTensor.to_cudf()` / `from_cudf()`. Needed to extract the raw flat values and offsets buffers from a cuDF list column.
- **cupy** (`cupy`): GPU array library. Already used in `StringTensor`. Bridges cuDF columns to torch tensors via DLPack (zero-copy).
- **sentence-transformers** (`sentence_transformers`): Already a runtime dependency. Loads the model and tokenizer. The model object contains a `Transformer` module (index 0) and a `Pooling` module (index 1) that can be called directly with `input_ids` and `attention_mask` dicts, bypassing `model.encode()`.
- **transformers** (`transformers`): HuggingFace library. Transitive dependency via sentence-transformers (always installed). Used only in `__init__` to inspect the tokenizer type — `PreTrainedTokenizerFast` exposes `.backend_tokenizer.model` to check if it's WordPiece.

### Key types

- **`WordPieceVocabulary`** (`cudf.core.wordpiece_tokenize`): Constructed from a `cudf.Series` of vocab entries. `.tokenize(series)` returns a `cudf.Series` with `ListDtype(int32)` — each row is a variable-length list of WordPiece token IDs.
- **`CharacterNormalizer`** (`cudf.core.character_normalizer`): BERT-style text normalization on GPU — lowercasing (when `do_lower=True`), accent stripping, punctuation separation. Special tokens like `[CLS]` are preserved during normalization. This step is required because `WordPieceVocabulary.tokenize()` expects pre-normalized input.
- **cuDF list column**: Internal layout is a flat contiguous values buffer + an offsets buffer (same layout as Arrow list arrays). `offsets[i]:offsets[i+1]` indexes into the flat buffer to get row `i`'s elements.

______________________________________________________________________

## Implementation

All changes are in `sdm/processing/text/sentence_transformer.py`.

### 1. `__init__`: detect tokenizer type and build GPU tokenizer objects

After loading the sentence-transformers model (which also loads the HuggingFace tokenizer), inspect the tokenizer to decide whether to use the GPU path:

```python
from transformers import PreTrainedTokenizerFast

tokenizer = model.tokenizer
is_wordpiece = (
    isinstance(tokenizer, PreTrainedTokenizerFast)
    and tokenizer.backend_tokenizer.model.__class__.__name__ == "WordPiece"
)
```

This is a type check and attribute read — no computation, no model loading.

If WordPiece and cuDF is available, build the GPU tokenizer objects once:

- **Vocabulary**: Extract the vocab from the HuggingFace tokenizer via `tokenizer.convert_ids_to_tokens(range(tokenizer.vocab_size))`, wrap in `cudf.Series`, pass to `WordPieceVocabulary(vocab_series)`.
- **Normalizer**: Construct `CharacterNormalizer(do_lower=tokenizer.do_lower_case, special_tokens=cudf.Series(list(tokenizer.all_special_tokens)))`. The `do_lower` flag matches the model — `bert-base-uncased` lowercases, `bert-base-cased` does not.
- **Special token IDs**: Store `tokenizer.cls_token_id`, `tokenizer.sep_token_id`, `tokenizer.pad_token_id`.
- **Max length**: Store `tokenizer.model_max_length`.

Store a flag (e.g., `self._use_gpu_tokenize`) to gate the path in `_transform`.

### 2. `_transform`: GPU tokenization path

When `_use_gpu_tokenize` is true, replace the current `to_arrow` → `to_pylist` → `model.encode()` flow:

#### 2a. StringTensor → cudf.Series

```python
text_series = text.to_cudf()
```

`StringTensor.to_cudf()` (in `sdm/tensor/string.py`) constructs a `cudf.Series` from the same GPU buffers the tensor already holds — zero-copy.

#### 2b. Handle nulls

Replace null strings with empty strings so the tokenizer receives valid input:

```python
if text.is_nullable:
    text_series = text_series.fillna("")
```

#### 2c. Normalize and tokenize

```python
normalized = self._normalizer.normalize(text_series)
token_lists = self._wpt.tokenize(normalized)
```

`token_lists` is a `cudf.Series` with `ListDtype(int32)`. Each row is a variable-length list of WordPiece token IDs. No special tokens (`[CLS]`, `[SEP]`) are added — that's our responsibility. No padding or truncation is applied.

#### 2d. Extract flat values + offsets as torch tensors

The cuDF list column internally stores a flat contiguous int32 buffer of all token IDs across all rows, plus an int32 offsets buffer of length N+1. Extract both as torch tensors on GPU via zero-copy DLPack/cupy bridge:

```python
list_col = token_lists._column
flat_values = torch.as_tensor(cupy.asarray(list_col.elements.values), device=device)
offsets_plc = list_col.plc_column.list_view().offsets()
offsets = torch.as_tensor(
    cupy.asarray(cudf.core.column.column.ColumnBase.create(offsets_plc).values),
    device=device,
)
```

At this point `flat_values` is `(total_tokens,)` and `offsets` is `(N+1,)`, both int32 torch tensors on GPU. `flat_values[offsets[i]:offsets[i+1]]` gives row `i`'s token IDs.

#### 2e. Scatter into padded tensor (truncate + special tokens + pad in one pass)

Build a single `(N, seq_len)` tensor that contains `[CLS] + truncated_tokens + [SEP] + padding` for each row:

```python
# Per-row token counts, clamped to max_length - 2 (reserving room for [CLS] and [SEP])
raw_lengths = offsets[1:] - offsets[:-1]
lengths = raw_lengths.clamp(max=max_length - 2)

# Dynamic padding: pad to the longest sequence in this batch, not to max_length globally
seq_len = int(lengths.max()) + 2

# Pre-fill with pad_id
input_ids = torch.full((N, seq_len), pad_id, device=device, dtype=torch.int32)

# [CLS] at position 0 for every row
input_ids[:, 0] = cls_id

# Build index arrays to scatter truncated token IDs into positions 1..lengths[i]+1
row_idx = torch.arange(N, device=device).repeat_interleave(lengths)
col_idx = torch.cat([torch.arange(l, device=device) for l in lengths]) + 1  # +1 for [CLS]
src_idx = torch.cat([
    torch.arange(offsets[i], offsets[i] + lengths[i], device=device)
    for i in range(N)
])
input_ids[row_idx, col_idx] = flat_values[src_idx]

# [SEP] right after the last real token in each row
input_ids[torch.arange(N, device=device), lengths + 1] = sep_id

# Attention mask: 1 for real tokens, 0 for padding
attention_mask = (input_ids != pad_id).to(torch.int32)
```

**Note**: The `torch.cat([torch.arange(...) for ...])` loops above are pseudocode to show the intent. The actual implementation must avoid Python loops. A vectorized formulation using `repeat_interleave` and cumulative sums:

```python
# Row index: each row i repeated lengths[i] times
row_idx = torch.arange(N, device=device).repeat_interleave(lengths)

# Column index: 0,1,...,lengths[0]-1, 0,1,...,lengths[1]-1, ... then +1 for [CLS]
col_idx = torch.arange(lengths.sum(), device=device) - torch.repeat_interleave(lengths.cumsum(0) - lengths, lengths) + 1

# Source index into flat_values: offsets[0]+0,...,offsets[0]+lengths[0]-1, offsets[1]+0,...
src_idx = torch.arange(lengths.sum(), device=device) - torch.repeat_interleave(lengths.cumsum(0) - lengths, lengths) + torch.repeat_interleave(offsets[:-1], lengths)

input_ids[row_idx, col_idx] = flat_values[src_idx]
```

#### 2f. Forward pass

Call the transformer and pooling modules directly instead of `model.encode()`:

```python
transformer = self._model.module[0]  # sentence_transformers Transformer module
pooling = self._model.module[1]      # sentence_transformers Pooling module

features = {"input_ids": input_ids, "attention_mask": attention_mask}
with torch.inference_mode():
    features = transformer(features)
    features = pooling(features)
emb = features["sentence_embedding"]
```

The sentence-transformers `Transformer` module accepts a dict with `input_ids` and `attention_mask` keys and returns a dict. The `Pooling` module takes the same dict and adds a `sentence_embedding` key.

This should be batched by `self.batch_size` — slice `input_ids` and `attention_mask` into chunks along dim 0, run each through the transformer + pooling, and concatenate the embeddings.

### 3. Fallback

When `_use_gpu_tokenize` is false (BPE model, or cuDF not available), use the existing `model.encode()` path unchanged.

Gate cuDF availability via `importlib.util.find_spec("cudf")` (same pattern used in `StringTensor` for the cuDF/Arrow backend selection).

______________________________________________________________________

## Future work

- GPU tokenization for BPE models via cuDF's `byte_pair_encode` API.
- Nested tensor path (`torch.nested.nested_tensor_from_jagged`) to eliminate padding entirely, using `flash_attn_varlen_func` or PyTorch nested tensor support in `TransformerEncoder`.
