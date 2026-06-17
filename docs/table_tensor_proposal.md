# TableTensor Proposal

## Summary

`TableTensor` should be a dense 2D `torch.Tensor` subclass with lightweight
schema metadata:

```text
TableTensor = tensorized scalar table + column names + semantic types
```

It is not a raw dataframe replacement. Raw, non-scalar modalities must be
encoded into scalar tensor columns before construction.

## Data Boundary

Use three distinct stages:

1. **Encoding**: raw dataframe/cuDF columns become scalar tensor columns.
2. **Preprocessing**: tensor-native recipe transforms operate on `TableTensor`.
3. **Packing**: context/query data becomes model-ready dense tensors.

This avoids a double preprocessing pipeline:

```text
raw df/cuDF
-> encode raw modalities
-> TableTensor
-> recipe preprocessing
-> PackedTableTensor / model input
```

Encoding makes raw columns tensor-representable. Preprocessing makes tensor
columns model-ready.

Examples:

- categorical strings -> ordinal ids
- timestamps -> relative/calendar scalar features
- multi-categorical columns -> hash/count/top-k scalar features
- text columns -> explicit embedding/hash/token-derived scalar features

If a raw modality cannot be safely represented as scalar columns without a
user-specified encoder, `to_tensor(...)` should raise instead of guessing.

## Schema

Keep schema minimal:

```python
class Stype(StrEnum):
    numerical = 'numerical'
    categorical = 'categorical'
    timestamp = 'timestamp'
    constant = 'constant'


@dataclass(frozen=True)
class Schema:
    names: tuple[str, ...]
    stypes: tuple[Stype, ...]
```

`Schema` should validate that names are unique and aligned with semantic types.
It should provide cheap structural helpers such as `index`, `select`, `drop`,
and `indices_for`.

## Tensor Subclass Strategy

Follow the same family of design as PyG `EdgeIndex`:

- create a wrapper subclass with `torch.Tensor._make_wrapper_subclass`;
- store the real tensor in `_data`;
- disable `__torch_function__`;
- implement `__torch_dispatch__`;
- preserve metadata only for a small whitelist of structural operations;
- unwrap to plain tensors for all other PyTorch operations;
- implement `__tensor_flatten__` and `__tensor_unflatten__`.

This gives native PyTorch ergonomics without pretending arbitrary tensor math
preserves table semantics.

Illustrative skeleton:

```python
class TableTensor(torch.Tensor):
    _data: torch.Tensor
    _schema: Schema
    _mask: torch.Tensor | None

    __torch_function__ = torch._C._disabled_torch_function_impl

    @staticmethod
    def __new__(
        cls,
        data: torch.Tensor,
        schema: Schema,
        mask: torch.Tensor | None = None,
    ) -> 'TableTensor':
        out = torch.Tensor._make_wrapper_subclass(
            cls,
            size=data.size(),
            strides=data.stride(),
            dtype=data.dtype,
            device=data.device,
            layout=data.layout,
            requires_grad=data.requires_grad,
        )
        out._data = data
        out._schema = schema
        out._mask = mask
        return out

    @property
    def values(self) -> torch.Tensor:
        return self._data
```

## Dispatch Contract

Only metadata-preserving operations should return `TableTensor`.

Initial whitelist:

- `clone`
- `detach`
- `to` / `_to_copy`
- `alias`
- row slicing
- structural column indexing when schema can be updated
- row-wise `cat` when schemas match exactly

Everything else should unwrap and return a plain `torch.Tensor`.

Examples:

```python
x.to('cuda')       # TableTensor
x.clone()          # TableTensor
x[:1024]           # TableTensor, row slice
x[:, 'target']     # TableTensor, one-dimensional single-column table
x[:, ['a', 'b']]   # TableTensor, two-dimensional table
x.drop('target')   # TableTensor, explicit table method

x[0]               # torch.Tensor
x + 1              # torch.Tensor
torch.log(x)       # torch.Tensor
x.mean(dim=0)      # torch.Tensor
```

This contract should be documented and tested directly.

## Table Methods

Prefer explicit table-aware methods over broad PyTorch overrides:

```python
x.column_index('fraud')
x.select(['age', 'income'])
x.drop('fraud')
x.indices_for(Stype.categorical)
x.pack(target='fraud', context=slice(0, 1024))
```

`pack(...)` should create the model-facing representation:

```text
features: context rows followed by query rows
y_context: labels for context rows
context_size: explicit context boundary
schema: model-facing feature schema
```

This matches TabICL/TabPFN-style execution while keeping the public context/query
boundary explicit.

## Encoding And Ensembling

Encoders are explicit and can later become an `encode=` phase in `Recipe`:

```python
Recipe(
    encode=[
        TimestampEncoder('created_at', ...),
        MultiCategoricalHashEncoder('genres', num_buckets=64),
    ],
    preprocess=[
        ImputeMissing(),
        QuantileTransform(apply_to={Stype.numerical}),
    ],
)
```

For v1, it is sufficient for `to_tensor(...)` to accept explicit encoders. If
ensembling over encoders becomes important, move those encoders into the recipe
while preserving the same logical boundary:

```text
encode: raw -> TableTensor
preprocess: TableTensor -> model-ready TableTensor
```

## Non-Goals

- Do not make `TableTensor` a dataframe replacement.
- Do not support raw text/list columns inside `TableTensor`.
- Do not preserve metadata through arbitrary tensor algebra.
- Do not build core preprocessing around pandas, NumPy, or sklearn.
- Do not hide major encoding choices behind defaults for text or
  multi-categorical data.
