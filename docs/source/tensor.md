# Table Semantics

Structured data models operate on heterogeneous tables across all kinds of data modalities: numerical values, categories, timestamps, free-form text, identifiers, and missing values.
Even within one modality, values may have different physical types, *e.g.*, strings, integers, or booleans can all represent categorical data.
Traditionally, converting such tables into numeric model inputs was left to the user.
The [`sdm.tensor`](api/tensor) package makes that conversion natural, PyTorch-native, and GPU-ready.
It provides a bridge from dataframe-like data to tensorized representations that preserve table semantics at the boundary while exposing tensor-shaped, device-aware objects inside the model.

The main entry point is the {py:class}`~sdm.tensor.TableTensor` class.
It is a **lossless**, **row-major tensor subclass** for structured data, in which columnar data is grouped in blocks by their [semantic type](api/generated/sdm.Stype) (*e.g.*, numerical, categorical, datetime, text).
The last dimension refers to the named column dimension `C`, while preceding dimensions can represent row, batch, or example dimensions.
This makes a table look like a tensor of shape `[..., C]` without flattening all columns into one dense array of a single data type.
In particular, it is

- **dataframe-like at the boundary:** contruct from [`pandas`](https://pandas.pydata.org/docs), [`arrow`](https://arrow.apache.org/docs), or [`cudf`](https://docs.rapids.ai/api/cudf) via zero-copy buffer views, and convert back when needed.
- **PyTorch-native in the middle:** use familar operations such as {py:meth}`~torch.Tensor.to`, {py:meth}`~torch.Tensor.view`, {py:meth}`~torch.Tensor.unsqueeze`, indexing, slicing, {py:func}`torch.cat`, and {py:func}`torch.stack`, with fast device movement and efficient transfer to accelerators.
- **lossless:** column names, semantic types, categorical vocabularies, string values, and missing-values are fully preserved.

## The Tensor Stack

The [`sdm.tensor`](api/tensor) package is layered from low-level tensor subclasses to full table schemas.
Each subclass supports multi-dimensional shapes and strides, and preserves standard PyTorch ergonomics (*e.g.*, zero-copy views and slicing):

- {py:class}`~sdm.tensor.VarLenTensor`: A tensor whose logical elements have variable-length payloads, *e.g.* for multi-categorical data.
- {py:class}`~sdm.tensor.StringTensor`: A specialized {py:class}`~sdm.tensor.VarLenTensor` for representing UTF-8 strings.
- {py:class}`~sdm.tensor.CategoricalTensor`: A tensor for representing categorical values via integer codes together with their mapping to original values.
  Category mappings can be ordinary {py:class}`torch.Tensor` instances or subclasses of it, *e.g.*, {py:class}`~sdm.tensor.StringTensor`.
- {py:class}`~sdm.tensor.TableTensor`: Combines the tensor types above into a lossless, table representation whose columns are grouped by semantic type.
  It is the main user-facing interface for model inputs and outputs.

## Working with {py:class}`~sdm.tensor.TableTensor`

A {py:class}`~sdm.tensor.TableTensor` currently supports the following semantic types:

- {py:attr}`~sdm.Stype.numerical`: continuous or real-valued inputs stored as a {py:class}`torch.Tensor`.
- {py:attr}`~sdm.Stype.categorical`: discrete values stored as a {py:class}`~sdm.tensor.CategoricalTensor`.
- {py:attr}`~sdm.Stype.datetime`: timestamps stored as an integer {py:class}`torch.Tensor` containing Unix timestamps in microseconds.
- {py:attr}`~sdm.Stype.text`: free-form strings stored as {py:class}`~sdm.tensor.StringTensor`.
- {py:attr}`~sdm.Stype.id`: identifier columns (*e.g.*, primary keys or foreign keys) stored as a {py:class}`~sdm.tensor.ColumnarTensor`.

```{figure} images/table_light.svg
:figclass: light-only
:width: 100%
```

```{figure} images/table_dark.svg
:figclass: dark-only
:width: 100%
```

A {py:class}`~sdm.tensor.TableTensor` can be created manually from tensor blocks or converted from [`pandas`](https://pandas.pydata.org/docs), [`arrow`](https://arrow.apache.org/docs), or [`cudf`](https://docs.rapids.ai/api/cudf) dataframes.
In addition to the dataframe itself, each column must be assigned a semantic type; use {py:func}`~sdm.infer_stypes` to infer semantic types automatically:

```python
import sdm
import torch
import pandas as pd

df = pd.DataFrame(
    {
        "age": [25, 31, 42],
        "income": [70_000.0, 105_000.0, 92_000.0],
        "country": ["US", "DE", "US"],
        "title": ["analyst", "engineer", "manager"],
        "user_id": [101, 102, 103],
    }
)

table = sdm.TableTensor.from_pandas(
    df,
    stypes={
        "age": "numerical",
        "income": "numerical",
        "country": "categorical",
        "title": "text",
        "user_id": "id",
    },
)

print(table.size())
# torch.Size([3, 5])

print(table.columns)
# {
#   Stype.numerical: ("age", "income"),
#   Stype.categorical: ("country",),
#   Stype.datetime: (),
#   Stype.text: ("title",),
#   Stype.id: ("user_id",),
# }

print(table.numerical.size())
# torch.Size([3, 2])

print(table.categorical.code)
# tensor([[0],
#         [1],
#         [0]])

print(table.categorical.categories[0].tolist())
# ["US", "DE"]
```

```{note}
The mapping from categories to integer codes is not a stable API guarantee and may depend on the input data or backend.
For fully deterministic category mappings, align categories at the processing level via {py:class}`~sdm.processing.categorical.AlignCategories`.
```

### Column Semantics

Because columns are stored in semantic blocks, {py:class}`~sdm.tensor.TableTensor` does not treat the original dataframe column order as a stable invariant.
Prefer column-name selection over positional assumptions.
Otherwise, PyTorch syntactic sugar is fully preserved.
Rows and batch dimensions behave like PyTorch tensor dimensions.
You can slice, index, reshape, stack, concatenate, and move tables between devices.

```python
print(table[["age", "country"]].size())
# torch.Size([3, 2])

print(table[:2].size())
# torch.Size([2, 5])

print(table.unsqueeze(0).expand(2, -1, -1).size())
# torch.Size([2, 3, 5])

print(torch.cat([table, table], dim=0).size())
# torch.Size([6, 5])

print(torch.cat([table[["age"]], table[["country"]]], dim=-1).size())
# torch.Size([3, 2])
```

However, operations that would destroy column semantics are rejected.
For example, raw integer indexing into the column dimension is not allowed:

```python
table[..., 0]  # RuntimeError: can't select the column dimension
```

Device movement follows the usual PyTorch style:

```python
print(table.to("cuda").device)
# cuda:0
```

### Round-Tripping

A {py:class}`~sdm.tensor.TableTensor` is intended to sit between dataframe interfaces and model code.
You can zero-copy back to [`pandas`](https://pandas.pydata.org/docs) or [`arrow`](https://arrow.apache.org/docs) when you want to leave the tensorized runtime:

```python
df = table.to_arrow()
df = table.to_pandas()
```

On CUDA, [`cudf`](https://docs.rapids.ai/api/cudf) can be used as the dataframe boundary:

```python
df = table.to_cudf()
```

```{note}
For CUDA workloads, using [``cudf``](https://docs.rapids.ai/api/cudf) is highly recommended.
Some table tensor operations are executed by exposing tensor buffers to [``cudf``](https://docs.rapids.ai/api/cudf).
Without it, these operations fall back to a CPU backend, which requires transferring data from the device to the host and back.
```

## Model Inputs And Outputs

A {py:class}`~sdm.tensor.TableTensor` acts as the primary abstraction for model inputs and outputs, and flows through GPU-accelerated preprocessing and ensembling.
You can learn more about data processing and model execution in the [Data Processing](processing) and [In-Context Learning Model Interface](icl) guides.
