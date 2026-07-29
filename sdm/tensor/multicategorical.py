from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, cast

import pyarrow as pa
import pyarrow.compute as pc
import torch
from torch import Tensor
from typing_extensions import Self, override

from sdm.tensor import CategoricalTensor, StringTensor, VarLenTensor
from sdm.tensor.io import arrow_as_tensor, to_arrow

if TYPE_CHECKING:
    import cudf

aten = torch.ops.aten

_NULL_LIST_CODE = -2


class MultiCategoricalTensor(CategoricalTensor):
    r"""A tensor for list-valued categorical columns.

    Category values are stored once per column, as in
    :class:`~sdm.tensor.CategoricalTensor`. Each logical element stores a
    variable-length list of category codes in a
    :class:`~sdm.tensor.VarLenTensor`.

    Raw codes use ``-1`` for a null category within a list. A singleton list
    containing only ``-2`` is reserved for a null outer list. Other negative
    codes and ``-2`` in any other position are invalid.

    Args:
        code: Variable-length categorical indices of shape ``[..., C]`` using
            the null encoding described above.
        categories: A tuple of ``C`` category vectors.
    """

    _code: VarLenTensor

    def __new__(
        cls,
        code: VarLenTensor,
        categories: Sequence[Tensor],
    ) -> Self:
        r"""Create a multi-categorical tensor wrapper."""
        if not isinstance(code, VarLenTensor):
            raise TypeError(
                f"Expected 'code' in {cls.__name__!r} to be a "
                f"'VarLenTensor' (got {code.__class__.__name__!r})"
            )
        return super().__new__(cls, code=code, categories=categories)

    @classmethod
    def from_arrow(
        cls,
        array: pa.Array | pa.ChunkedArray,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> Self:
        r"""Create a tensor from an Arrow list array.

        Args:
            array: A ``list`` or ``large_list`` Arrow array.
            dtype: The categorical code dtype.
            device: The device.
        """
        device = torch.device("cpu" if device is None else device)

        if isinstance(array, pa.ChunkedArray):
            if array.num_chunks == 1:
                array = array.chunk(0)
            else:
                array = array.combine_chunks()

        is_list = pa.types.is_list(array.type)
        is_large_list = pa.types.is_large_list(array.type)
        if not is_list and not is_large_list:
            raise TypeError(
                f"Expected 'array' in '{cls.__name__}.from_arrow' to have "
                f"'list' or 'large_list' type (got '{array.type}')"
            )

        values = array.values
        encoded = (
            values
            if pa.types.is_dictionary(values.type)
            else values.dictionary_encode()
        )
        arrow_code_type = pa.int64() if dtype == torch.int64 else pa.int32()
        indices = encoded.indices.fill_null(-1).cast(arrow_code_type)
        code_type = (
            pa.list_(arrow_code_type)
            if is_list
            else pa.large_list(arrow_code_type)
        )
        code_array = (
            pa.ListArray.from_arrays(array.offsets, indices)
            if is_list
            else pa.LargeListArray.from_arrays(array.offsets, indices)
        )
        if array.null_count > 0:
            code_array = pc.call_function(
                "if_else",
                [
                    pc.call_function("is_null", [array]),
                    pa.scalar([_NULL_LIST_CODE], type=code_type),
                    code_array,
                ],
            )

        code = VarLenTensor.from_arrow(code_array, device=device).unsqueeze(-1)
        if dtype is not None:
            code = cast(VarLenTensor, code.to(dtype=dtype))

        dictionary = encoded.dictionary
        if pa.types.is_string(dictionary.type) or pa.types.is_large_string(
            dictionary.type
        ):
            category = StringTensor.from_arrow(dictionary, device=device)
        elif pa.types.is_null(dictionary.type):
            category = torch.empty(0, dtype=torch.int64, device=device)
        else:
            category = arrow_as_tensor(dictionary, device=device)

        return cls(code=cast(VarLenTensor, code), categories=(category,))

    @classmethod
    def from_cudf(
        cls,
        ser: cudf.Series,
        *,
        dtype: torch.dtype = torch.int32,
        device: torch.device | str | None = None,
    ) -> Self:
        r"""Reject list-valued cuDF input until device-native IO is supported.

        Args:
            ser: The list-valued series.
            dtype: The categorical code dtype.
            device: The device.
        """
        raise NotImplementedError(
            "Creating a 'MultiCategoricalTensor' from cuDF is not yet "
            "supported"
        )

    @classmethod
    def from_tensor(cls, tensor: Tensor) -> Self:
        r"""Reject dense input, which cannot describe ragged category lists."""
        raise NotImplementedError(
            "Creating a 'MultiCategoricalTensor' from a dense tensor is not "
            "supported"
        )

    @override
    def to_arrow(self, names: Sequence[str] | None = None) -> pa.Table:
        r"""Convert this tensor to a two-dimensional Arrow table.

        Args:
            names: Column names.
        """
        if names is None:
            names = tuple(str(i) for i in range(self.size(-1)))
        elif len(names) != self.size(-1):
            raise ValueError(
                f"Expected 'names' to contain {self.size(-1)} entries "
                f"(got {len(names)})"
            )

        code_t = cast(
            VarLenTensor,
            self._code.movedim(-1, 0).contiguous().cpu(),
        )
        arrays: list[pa.Array] = []
        for value, category in zip(code_t.unbind(0), self.categories):
            code = cast(VarLenTensor, value)
            _validate_null_codes(code)
            data, offset = code.data_offset
            null_list = _null_list_mask(code).view(-1)
            indices = pa.array(
                data.numpy(),
                mask=(data < 0).numpy(),
            )
            values = pa.DictionaryArray.from_arrays(
                indices=indices,
                dictionary=to_arrow(category),
            )
            offsets = pa.array(offset.numpy())
            if offset.dtype == torch.int32:
                array = pa.ListArray.from_arrays(
                    offsets=offsets,
                    values=values,
                    mask=pa.array(null_list.numpy()),
                )
            else:
                array = pa.LargeListArray.from_arrays(
                    offsets=offsets,
                    values=values,
                    mask=pa.array(null_list.numpy()),
                )
            arrays.append(array)

        return pa.Table.from_arrays(arrays, names=names)

    @override
    def to_cudf(self, names: Sequence[str] | None = None) -> cudf.DataFrame:
        r"""Reject cuDF output until device-native list IO is supported.

        Args:
            names: Column names.
        """
        raise NotImplementedError(
            "Converting a 'MultiCategoricalTensor' to cuDF is not yet "
            "supported"
        )

    @property
    @override
    def code(self) -> VarLenTensor:
        r"""Return the variable-length categorical code tensor."""
        return self._code

    @classmethod
    def __torch_function__(
        cls,
        func: Any,
        types: tuple[type[Any], ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        if func is torch.isfinite or func is Tensor.isfinite:
            inp = cast(MultiCategoricalTensor, args[0])
            return ~_null_list_mask(inp._code)
        return super().__torch_function__(func, types, args, kwargs)

    @override
    def tolist(self) -> Any:
        def reshape(values: list[Any], size: tuple[int, ...]) -> Any:
            if len(size) == 0:
                return values[0]
            if len(size) == 1:
                return values
            step = math.prod(size[1:])
            return [
                reshape(values[i * step : (i + 1) * step], size[1:])
                for i in range(size[0])
            ]

        columns = tuple(self.to_arrow().to_pydict().values())
        rows = (
            [list(row) for row in zip(*columns)]
            if len(columns) > 0
            else [[] for _ in range(math.prod(self.size()[:-1]))]
        )
        return reshape(rows, tuple(self.size()[:-1]))


def _null_list_mask(code: VarLenTensor) -> Tensor:
    code = cast(VarLenTensor, code.contiguous())
    data, offset = code.data_offset
    counts = offset.diff()
    if data.numel() == 0:
        return torch.zeros(code.size(), dtype=torch.bool, device=code.device)

    first = data[offset[:-1].clamp(max=data.numel() - 1)]
    return ((counts == 1) & (first == _NULL_LIST_CODE)).view(code.size())


def _validate_null_codes(code: VarLenTensor) -> None:
    data, _ = code.data_offset
    invalid_negative = data < _NULL_LIST_CODE
    sentinel_count = (data == _NULL_LIST_CODE).sum()
    null_list_count = _null_list_mask(code).sum()
    if invalid_negative.any() or sentinel_count != null_list_count:
        raise ValueError(
            "Expected '-1' to encode null categories and a singleton '-2' "
            "to encode each null list"
        )


@MultiCategoricalTensor.implements(aten.isnan.default)
def _isnan(inp: MultiCategoricalTensor) -> Tensor:
    return _null_list_mask(inp.code)
