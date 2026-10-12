# TableTensor construction with torch.compile

This branch adds PyTorch tensor-subclass flattening and reconstruction protocols to `TableTensor`, `CategoricalTensor`, and `ColumnarTensor`. It addresses the constructor failure at `code.storage_offset()` without changing storage offsets or copying tensor data.

Public tensor properties are exposed as flattened children so compiled code can read `table.numerical` and `categorical.code`. Category vectors and heterogeneous columns are exposed individually. Empty ID containers are reconstructed from the table shape because they contain no tensor payload. Constructor arguments use semantic-type strings; the existing constructor still normalizes them to `Stype`.

```python
import torch
from sdm import TableTensor


def construct(value):
    table = TableTensor.from_tensor(value)
    return table, table.numerical + 1


compiled = torch.compile(construct, fullgraph=True, dynamic=True)
table, result = compiled(torch.randn(3, 2))
```

The change is separate from PR 1068 and is based on main. It does not change model fitting or normalization algorithms.

## CPU validation

All checks used the source on this branch. Native compiler checks used `fullgraph=True`, `dynamic=True`, and the Inductor backend.

The suite counts below include four compilation test cases used during validation and subsequently removed from the branch. The production implementation is unchanged.

| Check                                                     | PyTorch 2.7.1                               | PyTorch 2.14.0         |
| --------------------------------------------------------- | ------------------------------------------- | ---------------------- |
| Numerical table construction and numerical access         | 24/24 pass                                  | 24/24 pass             |
| Categorical construction and code access, native compiler | AOT metadata-printing failure               | 24/24 pass             |
| Categorical construction and code access, tracing backend | Pass                                        | Pass                   |
| Tensor test suite                                         | 152 passed, 49 skipped; 2 existing failures | 154 passed, 49 skipped |

The numerical checks cover FP32/FP64, row counts 3/5, widths 2/3, contiguous tensors, sliced offsets, and strided views. They preserve payload values, strides, and storage offsets. The two 2.7.1 test failures also reproduce on unchanged main: pandas conversions do not honor the `meta` default-device context.

The categorical 2.7.1 native failure occurs while AOT prints a fake categorical tensor in compiler metadata, raising `aten._local_scalar_dense.default`. It occurs with both static and dynamic shapes.

## Remaining limits

Complete preprocessing compilation is not established. In particular, full-graph `Standardize.fit_transform()` still encounters a separate compiler error when reading the `table.active_stypes` frozenset property. Its buffer-update fix remains separate in PR 1068. General schema operations, standalone empty `ColumnarTensor` compilation, and all nested nullable/string cases need further validation.

The extra relative-time tensor methods can be removed only after the existing table-based Standardize workflow has been validated end to end. Constructor support alone does not prove that workflow or complete model compilation.
