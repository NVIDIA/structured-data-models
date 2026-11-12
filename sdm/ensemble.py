# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import functools
import math
import operator
from collections.abc import Callable, Iterable, Iterator, Sequence
from typing import Any, Generic, Literal, Self, TypeAlias, cast, overload

import torch
from torch import Tensor
from torch.utils import _pytree
from typing_extensions import TypeVar

from sdm import Stype, StypeLike
from sdm.tensor import TableTensor
from sdm.tensor.categorical import _iter_categories
from sdm.tensor.mixin import DeviceMixin

T = TypeVar("T")
TensorT = TypeVar("TensorT", bound=Tensor, default=Tensor)


@functools.cache
@torch.compiler.assume_constant_result
def _operation_metadata(
    name: str, module: str, tensor_method: bool
) -> tuple[str, bool, bool, frozenset[torch.Tag]]:
    # Operator metadata is constant; Dynamo cannot trace packet.overloads().
    native = (
        tensor_method or module == "_operator" or module.startswith("torch")
    )
    table_operation = module == TableTensor.__module__
    name = {
        "truediv": "div",
        "and_": "bitwise_and",
        "or_": "bitwise_or",
        "invert": "bitwise_not",
    }.get(name, name)
    packet = getattr(torch.ops.aten, name, None) if native else None
    if packet is None:
        return name, native, table_operation, frozenset()
    return (
        name,
        native,
        table_operation,
        frozenset(
            tag
            for overload in packet.overloads()
            for tag in getattr(packet, overload).tags
        ),
    )


class EnsembleData(Generic[T]):
    """Represent an ordered collection of ensemble member values.

    Args:
        groups: Values containing the ensemble members.
        locations: Group index and position for each ensemble member.
    """

    _groups: tuple[T, ...]
    _locations: tuple[tuple[int, int], ...]

    def __init__(
        self,
        groups: Sequence[T],
        locations: Sequence[tuple[int, int]],
    ) -> None:
        self._groups = tuple(groups)
        self._locations = tuple(locations)
        self._validate()

    def _validate(self) -> None:
        group_sizes = tuple(self._group_size(group) for group in self._groups)
        actual_locations = set(self._locations)
        if len(actual_locations) == sum(group_sizes) and all(
            0 <= group_id < len(group_sizes)
            and 0 <= position < group_sizes[group_id]
            for group_id, position in actual_locations
        ):
            return

        # Detailed diagnostics only for invalid input.
        expected_locations = {
            (group_id, position)
            for group_id, size in enumerate(group_sizes)
            for position in range(size)
        }
        missing_locations = expected_locations - actual_locations
        if missing_locations:
            raise ValueError(
                "Expected 'locations' to cover all group positions "
                f"(missing {sorted(missing_locations)})"
            )

        invalid_locations = actual_locations - expected_locations
        if invalid_locations:
            raise ValueError(
                "Expected 'locations' to reference existing group positions "
                f"(got {sorted(invalid_locations)})"
            )

    def __len__(self) -> int:
        return len(self._locations)

    def __iter__(self) -> Iterator[T]:
        return (self[member] for member in range(len(self)))

    def __bool__(self) -> bool:
        raise RuntimeError(
            "Select an ensemble member before testing its truth value"
        )

    @overload
    def __getitem__(self, index: int) -> T: ...

    @overload
    def __getitem__(self, index: slice | Sequence[int]) -> Self: ...

    def __getitem__(self, index: int | slice | Sequence[int]) -> T | Self:
        if isinstance(index, int):
            group_id, position = self._locations[index]
            return self._select_member(self._groups[group_id], position)
        if isinstance(index, slice):
            index = range(*index.indices(len(self)))
        return self._select_members(index)

    def _iter_groups(self) -> Iterator[T]:
        return iter(self._groups)

    @staticmethod
    def _group_size(group: T) -> int:
        return 1

    @staticmethod
    def _select_member(group: T, position: int) -> T:
        return group

    def _select_members(self, member_ids: Sequence[int]) -> Self:
        locations = tuple(self._locations[index] for index in member_ids)
        group_ids = dict.fromkeys(group_id for group_id, _ in locations)
        remap = {group_id: index for index, group_id in enumerate(group_ids)}
        return self.__class__(
            groups=tuple(self._groups[group_id] for group_id in group_ids),
            locations=tuple((remap[group_id], 0) for group_id, _ in locations),
        )

    @staticmethod
    def execute(
        function: Callable[..., Any],
        /,
        *args: Any,
        batched: bool = False,
        randomness: Literal["same", "different"] = "same",
        **kwargs: Any,
    ) -> Any:
        """Apply an operation to aligned members, retaining shared results.

        Tensor operations run in physical batches with logical tensor axes.
        Python objects and tensor subclasses without batching support execute
        once per distinct combination of member values.

        Args:
            function: Operation applied to corresponding member values.
            *args: Arguments, optionally containing ensemble values.
            batched: Whether the operation already preserves an additional
                leading batch dimension. Otherwise tensor axes are logical.
            randomness: Whether shared inputs draw independently per member.
            **kwargs: Keyword arguments, optionally containing ensemble values.
        """
        if any(
            isinstance(value, (tuple, list, dict))
            for value in (*args, *kwargs.values())
        ):
            leaves, arguments = _pytree.tree_flatten((args, kwargs))
        else:
            leaves, arguments = (*args, *kwargs.values()), None
        ensemble_inputs = tuple(
            {id(x): x for x in leaves if isinstance(x, EnsembleData)}.values()
        )
        if not ensemble_inputs:
            return function(*args, **kwargs)
        num_members = len(ensemble_inputs[0])
        if any(len(value) != num_members for value in ensemble_inputs[1:]):
            raise ValueError(
                "Ensemble inputs must have the same number of members"
            )
        operation, native, table_operation, operator_tags = (
            _operation_metadata(
                getattr(function, "__name__", ""),
                getattr(function, "__module__", "") or "",
                isinstance(function, type(Tensor.mean)),
            )
        )
        table_operation |= function is torch.cat and all(
            isinstance(value, EnsembleTable) for value in ensemble_inputs
        )
        physical = batched or table_operation
        tags = set(getattr(function, "tags", ()))
        tags.update(operator_tags)
        pointwise = torch.Tag.pointwise in tags
        random = (
            randomness == "different"
            or torch.Tag.nondeterministic_seeded in tags
        )
        shared = not random and all(
            len(value._groups) == 1
            and value._group_size(value._groups[0]) == 1
            for value in ensemble_inputs
        )
        if shared:
            # One shared value needs no alignment, batch selection or vmap.
            physical &= any(
                isinstance(value, EnsembleTensor) for value in ensemble_inputs
            )
            flat = []
            for value in leaves:
                if physical and isinstance(value, EnsembleTensor):
                    value = value._groups[0]
                elif isinstance(value, EnsembleData):
                    value = value[0]
                elif physical and isinstance(value, Tensor) and value.ndim:
                    value = value.unsqueeze(0)
                flat.append(value)
            if arguments is None:
                positional, keywords = (
                    flat[: len(args)],
                    dict(zip(kwargs, flat[len(args) :])),
                )
            else:
                positional, keywords = _pytree.tree_unflatten(flat, arguments)
            result = function(*positional, **keywords)
            outputs, structure = _pytree.tree_flatten(result)
            for index, output in enumerate(outputs):
                if not isinstance(output, Tensor):
                    continue
                source = next(
                    (
                        original
                        for original, member in zip(leaves, flat)
                        if isinstance(original, EnsembleTensor)
                        and output is member
                    ),
                    None,
                )
                if source is not None:
                    outputs[index] = source
                    continue
                cls = (
                    EnsembleTable
                    if isinstance(output, TableTensor)
                    else EnsembleTensor
                )
                outputs[index] = cls(
                    groups=(output if physical else output.unsqueeze(0),),
                    locations=((0, 0),) * num_members,
                )
            return _pytree.tree_unflatten(outputs, structure)
        destination = kwargs.get("out")
        if destination is None and (
            (
                native
                and operation.endswith("_")
                and not operation.startswith("__")
            )
            or function is operator.setitem
        ):
            destination = args[0]
        if (
            isinstance(destination, EnsembleData)
            and len(set(destination._locations)) < num_members
        ):
            writes = {}
            for member in range(num_members):
                location = destination._locations[member]
                binding = tuple(
                    value._locations[member] for value in ensemble_inputs
                )
                if location in writes and (
                    random or writes[location] != binding
                ):
                    raise RuntimeError(
                        "Cannot write different member values into "
                        "shared ensemble storage"
                    )
                writes[location] = binding

        # Keep the largest input in place; gathering small fitted statistics is
        # cheaper than materializing the full table to match their grouping.
        tensors = tuple(
            value
            for value in ensemble_inputs
            if isinstance(value, EnsembleTensor)
        )
        anchor = (
            max(
                tensors,
                key=lambda x: sum(group.numel() for group in x._groups),
            )
            if tensors
            else ensemble_inputs[0]
        )
        if isinstance(destination, EnsembleData):
            anchor = destination
        partitions: dict[tuple[Any, ...], list[int]] = {}
        raw_output = isinstance(destination, Tensor)
        if raw_output and num_members > 1:
            raise RuntimeError(
                "Use an ensemble output buffer for member-specific results"
            )
        for member in range(num_members):
            key: list[Any] = [anchor._locations[member][0]]
            if random:
                key.append(member)
            for value in ensemble_inputs:
                group_id, position = value._locations[member]
                group = value._groups[group_id]
                if isinstance(group, Tensor) and type(group) is Tensor:
                    key.append((group.shape[1:], group.dtype, group.device))
                elif isinstance(group, Tensor):
                    key.append(("tensor_subclass", group_id))
                else:
                    try:
                        hash(group)
                    except TypeError:
                        key.append((id(group),))
                    else:
                        key.append((type(group), group))
            partitions.setdefault(tuple(key), []).append(member)

        results: list[Any] = []
        locations = [(0, 0)] * num_members
        for members in partitions.values():
            if isinstance(destination, EnsembleData):
                members.sort(key=lambda member: destination._locations[member])
            # Aliased logical members need only one physical calculation.
            bindings: dict[tuple[Any, ...], int] = {}
            representatives: list[int] = []
            for member in members:
                binding = tuple(
                    value._locations[member] for value in ensemble_inputs
                )
                if binding not in bindings:
                    bindings[binding] = len(representatives)
                    representatives.append(member)

            selected: dict[int, Any] = {}
            batch_dims: dict[int, int | None] = {}
            batchable = (
                (len(representatives) > 1 or physical or pointwise)
                and (native or physical)
                and not raw_output
                and operation
                not in {
                    "size",
                    "dim",
                    "numel",
                    "stride",
                    "element_size",
                    "get_device",
                    "nonzero",
                    "masked_select",
                    "unique",
                    "unique_consecutive",
                    "getitem",
                    "setitem",
                    "item",
                    "tolist",
                    "equal",
                    "allclose",
                    "type",
                    "is_contiguous",
                    "is_floating_point",
                    "is_complex",
                    "is_inference",
                    "data_ptr",
                    "rand",
                    "randn",
                    "randint",
                    "randperm",
                    "rand_like",
                    "randn_like",
                }
            )
            for value in ensemble_inputs:
                if not batchable:
                    break
                source = tuple(
                    value._locations[member] for member in representatives
                )
                group_ids = {group_id for group_id, _ in source}
                if not isinstance(value, EnsembleTensor):
                    selected[id(value)] = value[representatives[0]]
                    batch_dims[id(value)] = None
                    continue
                if physical and all(
                    value._groups[group_id].numel() == 0
                    for group_id in group_ids
                ):
                    # Empty blocks carry shape/dtype metadata, not distinct
                    # member values. Reuse one empty view for the whole batch.
                    group = value._groups[source[0][0]]
                    selected[id(value)] = group[:1].expand(
                        len(representatives), *group.shape[1:]
                    )
                    batch_dims[id(value)] = 0
                elif len(set(source)) == 1:
                    group_id, position = source[0]
                    group = value._groups[group_id]
                    if (physical or pointwise) and isinstance(group, Tensor):
                        selected[id(value)] = (
                            group
                            if group.size(0) == 1
                            else group[position : position + 1]
                        )
                        if len(representatives) > 1:
                            selected[id(value)] = selected[id(value)].expand(
                                len(representatives), *group.shape[1:]
                            )
                        batch_dims[id(value)] = 0
                    else:
                        selected[id(value)] = value[representatives[0]]
                        batch_dims[id(value)] = None
                elif all(
                    type(value._groups[group_id]) is Tensor
                    or (
                        physical
                        and isinstance(value._groups[group_id], TableTensor)
                    )
                    for group_id in group_ids
                ):
                    positions = tuple(position for _, position in source)
                    start = positions[0]
                    if len(group_ids) == 1 and positions == tuple(
                        range(start, start + len(positions))
                    ):
                        group = value._groups[source[0][0]]
                        selected[id(value)] = (
                            group
                            if start == 0 and len(positions) == group.size(0)
                            else group[start : start + len(positions)]
                        )
                    else:
                        if value is destination:
                            batchable = False
                            break
                        selected[id(value)] = torch.stack(
                            [value[member] for member in representatives]
                        )
                    batch_dims[id(value)] = 0
                else:
                    batchable = False
                if not physical and any(
                    type(value._groups[group_id]) is not Tensor
                    for group_id in group_ids
                ):
                    batchable = False

            batchable &= any(
                isinstance(value, Tensor) for value in selected.values()
            )

            if batchable:
                flat = []
                for value in leaves:
                    if isinstance(value, EnsembleData):
                        value = selected[id(value)]
                    elif physical and isinstance(value, Tensor) and value.ndim:
                        # Plain tensors are shared logical values, not batches.
                        value = value.unsqueeze(0).expand(
                            len(representatives), *value.shape
                        )
                    flat.append(value)
                dims = tuple(
                    batch_dims[id(value)]
                    if isinstance(value, EnsembleData)
                    else None
                    for value in leaves
                )
                if pointwise:
                    rank = max(
                        value.ndim - (dim == 0)
                        for value, dim in zip(flat, dims)
                        if isinstance(value, Tensor)
                    )
                    scalar = any(
                        isinstance(value, Tensor) and value.ndim == (dim == 0)
                        for value, dim in zip(flat, dims)
                    )
                    if (
                        scalar
                        and rank
                        and len(
                            {
                                value.dtype
                                for value in flat
                                if isinstance(value, Tensor)
                            }
                        )
                        > 1
                    ):
                        # vmap retains zero-dimensional scalar type promotion.
                        pointwise = False
                    else:
                        for index, (value, dim) in enumerate(zip(flat, dims)):
                            if (
                                isinstance(value, Tensor)
                                and dim == 0
                                and value.ndim < rank + 1
                            ):
                                flat[index] = value.reshape(
                                    (value.size(0),)
                                    + (1,) * (rank + 1 - value.ndim)
                                    + tuple(value.shape[1:])
                                )

                # vmap hides the physical ensemble axis from reductions, views,
                # broadcasting and indexing; no per-operation axis registry.
                def invoke(*values: Any) -> Any:
                    if arguments is None:
                        positional, keywords = (
                            values[: len(args)],
                            dict(zip(kwargs, values[len(args) :])),
                        )
                    else:
                        positional, keywords = _pytree.tree_unflatten(
                            values, arguments
                        )
                    return function(*positional, **keywords)

                result = (
                    invoke(*flat)
                    if physical or pointwise
                    else torch.vmap(invoke, in_dims=dims)(*flat)
                )
                output_id = len(results)
                results.append(result)
                for member in members:
                    binding = tuple(
                        value._locations[member] for value in ensemble_inputs
                    )
                    locations[member] = (output_id, bindings[binding])
            else:
                for position, member in enumerate(representatives):
                    flat = [
                        value[member]
                        if isinstance(value, EnsembleData)
                        else value
                        for value in leaves
                    ]
                    if arguments is None:
                        positional, keywords = (
                            flat[: len(args)],
                            dict(zip(kwargs, flat[len(args) :])),
                        )
                    else:
                        positional, keywords = _pytree.tree_unflatten(
                            flat, arguments
                        )
                    result = function(*positional, **keywords)
                    result = (
                        result.unsqueeze(0)
                        if isinstance(result, Tensor)
                        else _pytree.tree_map(
                            lambda x: (
                                x.unsqueeze(0) if isinstance(x, Tensor) else x
                            ),
                            result,
                        )
                    )
                    output_id = len(results)
                    results.append(result)
                    for other in members:
                        binding = tuple(
                            value._locations[other]
                            for value in ensemble_inputs
                        )
                        if bindings[binding] == position:
                            locations[other] = (output_id, 0)

        if isinstance(results[0], Tensor):
            if isinstance(destination, EnsembleTensor):
                return destination
            cls = (
                EnsembleTable
                if isinstance(results[0], TableTensor)
                else EnsembleTensor
            )
            return cls(groups=results, locations=locations)
        first, structure = _pytree.tree_flatten(results[0])
        if not any(isinstance(value, Tensor) for value in first):
            example = results[0]
            if all(value == example for value in results):
                return example
            return EnsembleData(
                groups=results,
                locations=[(group_id, 0) for group_id, _ in locations],
            )
        outputs = [_pytree.tree_flatten(result)[0] for result in results]
        reconstructed: list[Any] = []
        for index, example in enumerate(first):
            groups = tuple(output[index] for output in outputs)
            if isinstance(example, Tensor):
                cls = (
                    EnsembleTable
                    if isinstance(example, TableTensor)
                    else EnsembleTensor
                )
                reconstructed.append(cls(groups=groups, locations=locations))
            elif all(
                value is example
                or (
                    isinstance(value, (str, int, float, bool, type(None)))
                    and value == example
                )
                for value in groups
            ):
                reconstructed.append(example)
            else:
                reconstructed.append(
                    EnsembleData(
                        groups=groups,
                        locations=[(group_id, 0) for group_id, _ in locations],
                    )
                )
        return _pytree.tree_unflatten(reconstructed, structure)


class EnsembleTensor(EnsembleData[TensorT]):
    """Store and group tensors as an ensemble.

    Args:
        groups: Tensors containing the ensemble members.
        locations: Group index and position for each ensemble member.
    """

    __hash__ = object.__hash__

    @classmethod
    def __torch_function__(
        cls,
        function: Callable[..., Any],
        types: tuple[type, ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        return EnsembleData.execute(function, *args, **(kwargs or {}))

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_") or not self._groups:
            raise AttributeError(name)
        attribute = getattr(type(self._groups[0]), name)
        if callable(attribute):
            return functools.partial(EnsembleData.execute, attribute, self)
        if name in {
            "dtype",
            "device",
            "layout",
            "requires_grad",
            "is_cuda",
            "is_cpu",
            "itemsize",
        }:
            values = tuple(getattr(group, name) for group in self._groups)
            if all(value == values[0] for value in values[1:]):
                return values[0]
            return EnsembleData(
                groups=values,
                locations=tuple((group, 0) for group, _ in self._locations),
            )
        return EnsembleData.execute(getattr, self, name)

    @property
    def shape(self) -> tuple[int, ...] | EnsembleData:
        """Member tensor shapes, excluding the physical ensemble axis."""
        return self.size()

    @property
    def ndim(self) -> int | EnsembleData:
        """Number of logical dimensions in each member tensor."""
        return self.dim()

    def size(self, dim: int | None = None) -> Any:
        """Return member sizes along a logical dimension, or all dimensions."""
        values = tuple(
            tuple(group.shape[1:]) if dim is None else group.shape[1:][dim]
            for group in self._groups
        )
        if all(value == values[0] for value in values[1:]):
            return values[0]
        return EnsembleData(
            groups=values,
            locations=tuple((group, 0) for group, _ in self._locations),
        )

    def dim(self) -> int | EnsembleData:
        """Return the number of logical dimensions in each member."""
        return EnsembleData.execute(len, self.size())

    def numel(self) -> int | EnsembleData:
        """Return the number of elements in each member tensor."""
        return EnsembleData.execute(math.prod, self.size())

    def as_tensor(self) -> TensorT:
        """Expose logical members along a leading tensor dimension.

        Reuse a physical batch or a shared expanded view when possible;
        otherwise stack the members in logical order.
        """
        if len(self._groups) == 1:
            group = self._groups[0]
            if self._locations == tuple((0, i) for i in range(len(self))):
                return group
            if group.size(0) == 1:
                return cast(TensorT, group.expand(len(self), *group.shape[1:]))
        return cast(TensorT, torch.stack(tuple(self)))

    def mean(
        self,
        dim: int | tuple[int, ...] | None = None,
        keepdim: bool = False,
        *,
        dtype: torch.dtype | None = None,
    ) -> EnsembleTensor:
        """Return each member's mean along logical tensor dimensions."""
        return EnsembleData.execute(
            torch.mean, self, dim=dim, keepdim=keepdim, dtype=dtype
        )

    def __add__(self, other: Any) -> EnsembleTensor:
        return EnsembleData.execute(operator.add, self, other)

    __radd__ = __add__

    def __sub__(self, other: Any) -> EnsembleTensor:
        return EnsembleData.execute(operator.sub, self, other)

    def __rsub__(self, other: Any) -> EnsembleTensor:
        return EnsembleData.execute(operator.sub, other, self)

    def __mul__(self, other: Any) -> EnsembleTensor:
        return EnsembleData.execute(operator.mul, self, other)

    __rmul__ = __mul__

    def __matmul__(self, other: Any) -> EnsembleTensor:
        return EnsembleData.execute(operator.matmul, self, other)

    def __rmatmul__(self, other: Any) -> EnsembleTensor:
        return EnsembleData.execute(operator.matmul, other, self)

    def __truediv__(self, other: Any) -> EnsembleTensor:
        return EnsembleData.execute(operator.truediv, self, other)

    def __rtruediv__(self, other: Any) -> EnsembleTensor:
        return EnsembleData.execute(operator.truediv, other, self)

    def __iadd__(self, other: Any) -> Self:
        return self.add_(other)

    def __isub__(self, other: Any) -> Self:
        return self.sub_(other)

    def __imul__(self, other: Any) -> Self:
        return self.mul_(other)

    def __itruediv__(self, other: Any) -> Self:
        return self.div_(other)

    def __iand__(self, other: Any) -> Self:
        return self.bitwise_and_(other)

    def __ior__(self, other: Any) -> Self:
        return self.bitwise_or_(other)

    def __pow__(self, other: Any) -> EnsembleTensor:
        return EnsembleData.execute(operator.pow, self, other)

    def __rpow__(self, other: Any) -> EnsembleTensor:
        return EnsembleData.execute(operator.pow, other, self)

    def __neg__(self) -> EnsembleTensor:
        return EnsembleData.execute(operator.neg, self)

    def __invert__(self) -> EnsembleTensor:
        return EnsembleData.execute(operator.invert, self)

    def __eq__(self, other: Any) -> EnsembleTensor:  # ty: ignore[invalid-method-override]
        return EnsembleData.execute(operator.eq, self, other)

    def __ne__(self, other: Any) -> EnsembleTensor:  # ty: ignore[invalid-method-override]
        return EnsembleData.execute(operator.ne, self, other)

    def __lt__(self, other: Any) -> EnsembleTensor:
        return EnsembleData.execute(operator.lt, self, other)

    def __le__(self, other: Any) -> EnsembleTensor:
        return EnsembleData.execute(operator.le, self, other)

    def __gt__(self, other: Any) -> EnsembleTensor:
        return EnsembleData.execute(operator.gt, self, other)

    def __ge__(self, other: Any) -> EnsembleTensor:
        return EnsembleData.execute(operator.ge, self, other)

    def __and__(self, other: Any) -> EnsembleTensor:
        return EnsembleData.execute(operator.and_, self, other)

    def __or__(self, other: Any) -> EnsembleTensor:
        return EnsembleData.execute(operator.or_, self, other)

    def __setitem__(self, index: Any, value: Any) -> None:
        EnsembleData.execute(operator.setitem, self, index, value)

    def __getitem__(self, index: Any) -> Any:
        if isinstance(index, (int, slice, list, range)) or (
            isinstance(index, tuple)
            and all(isinstance(value, int) for value in index)
        ):
            return super().__getitem__(index)
        return EnsembleData.execute(operator.getitem, self, index)

    @staticmethod
    def _group_size(group: TensorT) -> int:
        return group.size(0)

    @staticmethod
    def _select_member(
        group: TensorT,
        position: int,
    ) -> TensorT:
        return cast(TensorT, group[position])

    def _select_members(self, member_ids: Sequence[int]) -> Self:
        member_ids = tuple(member_ids)
        if member_ids == tuple(range(len(self))):
            return self

        locations = tuple(
            self._locations[member_id] for member_id in member_ids
        )
        positions_by_group: dict[int, dict[int, int]] = {}
        for group_id, position in locations:
            positions = positions_by_group.setdefault(group_id, {})
            positions.setdefault(position, len(positions))

        groups: list[TensorT] = []
        group_ids: dict[int, int] = {}
        for new_group_id, (group_id, positions) in enumerate(
            positions_by_group.items()
        ):
            group = self._groups[group_id]
            selected_positions = tuple(positions)
            group_ids[group_id] = new_group_id
            start = selected_positions[0]
            step = (
                selected_positions[1] - start
                if len(selected_positions) > 1
                else 1
            )
            stop = start + step * len(selected_positions)
            if selected_positions == tuple(range(group.size(0))):
                groups.append(group)
            elif step > 0 and selected_positions == tuple(
                range(start, stop, step)
            ):
                # Evenly spaced members are selected as a view, not a copy.
                groups.append(cast(TensorT, group[start:stop:step]))
            else:
                groups.append(
                    cast(
                        TensorT,
                        torch.stack(
                            [
                                group[position]
                                for position in selected_positions
                            ],
                            dim=0,
                        ),
                    )
                )

        return self.__class__(
            groups=groups,
            locations=tuple(
                (group_ids[group_id], positions_by_group[group_id][position])
                for group_id, position in locations
            ),
        )


class EnsembleTable(DeviceMixin, EnsembleTensor[TableTensor]):
    """Store and group input :class:`~sdm.tensor.TableTensor` as an ensemble.

    Each ensemble member is associated with one
    :class:`~sdm.tensor.TableTensor`.
    Shared tables are stored only once. Compatible tables are stacked along a
    leading dimension and form a group.

    Args:
        groups: Sequence of :class:`~sdm.tensor.TableTensor`, optionally
            stacked along their leading dimension.
        locations: ``(group, batch)`` location of each ensemble member.
    """

    @property
    def active_stypes(self) -> tuple[Stype, ...]:
        """Semantic types present in at least one member."""
        return tuple(
            dict.fromkeys(
                stype
                for group in self._groups
                for stype, columns in group.column_groups
                if columns
            )
        )

    @property
    def numerical(self) -> EnsembleTensor:
        """Return numerical blocks, preserving sharing and physical batches."""
        return EnsembleTensor(
            groups=tuple(group.numerical for group in self._groups),
            locations=self._locations,
        )

    @property
    def categorical(self) -> EnsembleTensor:
        """Return categorical blocks, preserving sharing and batches."""
        return EnsembleTensor(
            groups=tuple(group.categorical for group in self._groups),
            locations=self._locations,
        )

    @property
    def text(self) -> EnsembleTensor:
        """Text blocks with their original sharing and physical batches."""
        return EnsembleTensor(
            groups=tuple(group.text for group in self._groups),
            locations=self._locations,
        )

    @property
    def datetime(self) -> EnsembleTensor:
        """Datetime blocks with their original sharing and physical batches."""
        return EnsembleTensor(
            groups=tuple(group.datetime for group in self._groups),
            locations=self._locations,
        )

    @property
    def id(self) -> EnsembleTensor:
        """Return identifier blocks, preserving sharing and batches."""
        return EnsembleTensor(
            groups=tuple(group.id for group in self._groups),
            locations=self._locations,
        )

    def replace_blocks(
        self,
        *,
        numerical: Tensor | EnsembleTensor | None = None,
        categorical: Tensor | EnsembleTensor | None = None,
        datetime: Tensor | EnsembleTensor | None = None,
        text: Tensor | EnsembleTensor | None = None,
        id: Tensor | EnsembleTensor | None = None,
    ) -> EnsembleTable:
        """Replace semantic blocks, preserving each member's column schema.

        Args:
            numerical: Replacement numerical values.
            categorical: Replacement categorical values and vocabularies.
            datetime: Replacement datetime values.
            text: Replacement text values.
            id: Replacement identifier values.
        """
        blocks: dict[str, Any] = {}
        for name, replacement in (
            ("numerical", numerical),
            ("categorical", categorical),
            ("datetime", datetime),
            ("text", text),
            ("id", id),
        ):
            block = getattr(self, name) if replacement is None else replacement
            if isinstance(block, Tensor):
                block = EnsembleTensor(
                    groups=(block.unsqueeze(0),),
                    locations=((0, 0),) * len(self),
                )
            blocks[name] = block
        # Do not align/materialize old values that are being replaced.
        columns = self.columns
        constructor = TableTensor
        if isinstance(columns, EnsembleData):
            blocks["columns"] = columns
        else:
            constructor = functools.partial(TableTensor, columns=columns)
        return EnsembleData.execute(constructor, **blocks, batched=True)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        attribute = getattr(TableTensor, name)
        if callable(attribute):
            return functools.partial(EnsembleData.execute, attribute, self)
        if name in {
            "dtype",
            "column_groups",
            "columns",
            "column_names",
            "numerical_columns",
            "categorical_columns",
            "text_columns",
            "datetime_columns",
            "id_columns",
            "stypes",
        }:
            values = tuple(getattr(group, name) for group in self._groups)
            if all(value == values[0] for value in values[1:]):
                return values[0]
            return EnsembleData(
                groups=values,
                locations=tuple((group, 0) for group, _ in self._locations),
            )
        return EnsembleData.execute(getattr, self, name)

    @classmethod
    def from_table(
        cls,
        table: TableTensor,
        *,
        num_members: int,
    ) -> Self:
        """Create an ensemble, sharing one table across all members.

        Args:
            table: Table used across members.
            num_members: Positive number of ensemble members.

        Returns:
            An :class:`~sdm.EnsembleTable` with one group.
        """
        return cls(
            groups=(cast(TableTensor, table.unsqueeze(0)),),
            locations=((0, 0),) * num_members,
        )

    @classmethod
    def from_tables(
        cls,
        tables: Sequence[TableTensor],
        member_table_ids: Sequence[int],
    ) -> Self:
        """Create an ensemble table from tables and their member assignments.

        Each referenced input table forms a separate group.
        ``member_table_ids`` contains one table index per member. For
        example, ``(0, 1, 0)`` assigns the first table to members 0 and 2 and
        the second table to member 1.

        Args:
            tables: Tables available to the ensemble members.
            member_table_ids: Index into ``tables`` for each ensemble member.

        Returns:
            An ensemble table preserving member order.
        """
        referenced_table_ids = set(member_table_ids)
        table_ids = tuple(
            table_id
            for table_id in range(len(tables))
            if table_id in referenced_table_ids
        )
        group_ids = {
            table_id: group_id for group_id, table_id in enumerate(table_ids)
        }
        return cls(
            groups=tuple(
                cast(TableTensor, tables[table_id].unsqueeze(0))
                for table_id in table_ids
            ),
            locations=tuple(
                (group_ids[table_id], 0) for table_id in member_table_ids
            ),
        )

    @staticmethod
    def _pack_tables(
        tables: Sequence[TableTensor],
    ) -> tuple[tuple[TableTensor, ...], tuple[tuple[int, int], ...]]:
        compatible_groups: dict[tuple[object, ...], list[int]] = {}
        compatibility_keys: list[tuple[object, ...]] = []
        vocabularies: list[tuple[Tensor, ...]] = []
        for index, table in enumerate(tables):
            if torch.compiler.is_compiling():
                # Compilation: id() can fail or give one tensor different IDs.
                categories = tuple(_iter_categories(table.categorical))
                for vocabulary_id, other in enumerate(vocabularies):
                    if len(categories) == len(other) and all(
                        category is other_category
                        for category, other_category in zip(categories, other)
                    ):
                        vocabulary_key = (vocabulary_id,)
                        break
                else:
                    # No tuple matched; the next list index is its new ID.
                    vocabulary_key = (len(vocabularies),)
                    vocabularies.append(categories)
            else:
                # Eager execution: use id() directly to avoid the search.
                vocabulary_key = tuple(
                    id(category) for category in table.categorical.categories
                )
            # Shape, schema, block layout, device, and categorical vocabularies
            # must match for torch.stack to preserve member semantics.
            compatibility_key: tuple[object, ...] = (
                table.column_groups,
                table.device,
                vocabulary_key,
                tuple(
                    (
                        stype,
                        type(block),
                        block.layout,
                        block.dtype,
                        block.size(),
                    )
                    for stype, block in TableTensor.items(table)
                ),
            )
            if torch.compiler.is_compiling():
                # Hash group IDs to avoid fixing row and column counts during
                # compilation; compare full keys to preserve grouping.
                for key_id, key in enumerate(compatibility_keys):
                    if key == compatibility_key:
                        compatibility_key = (key_id,)
                        break
                else:
                    compatibility_keys.append(compatibility_key)
                    compatibility_key = (len(compatibility_keys) - 1,)
            compatible_groups.setdefault(compatibility_key, []).append(index)

        groups: list[TableTensor] = []
        locations = [(-1, -1)] * len(tables)
        for indices in compatible_groups.values():
            group_index = len(groups)
            groups.append(
                cast(TableTensor, tables[indices[0]].unsqueeze(0))
                if len(indices) == 1
                else cast(
                    TableTensor,
                    torch.stack(
                        tensors=[tables[index] for index in indices],
                        dim=0,
                    ),
                )
            )
            for position, index in enumerate(indices):
                locations[index] = (group_index, position)

        return tuple(groups), tuple(locations)

    def gather_members(
        self,
        tables: Sequence[Self],
        member_ids: Sequence[int],
    ) -> Self:
        """Gather members from multiple ensemble tables.

        ``tables[i][member_ids[i]]`` supplies output member ``i``.

        Args:
            tables: Source ensemble table for each output member.
            member_ids: Logical source member position for each output member.

        Returns:
            An ensemble table in logical member order.
        """
        if len(tables) != len(member_ids):
            raise ValueError("Expected one source member per ensemble table")

        outputs: list[TableTensor] = []
        output_id_by_source: dict[tuple[int, tuple[int, int]], int] = {}
        member_table_ids = []
        for table, member_id in zip(tables, member_ids, strict=True):
            location = table._locations[member_id]
            key = (id(table), location)
            output_id = output_id_by_source.get(key)
            if output_id is None:
                output_id = len(outputs)
                output_id_by_source[key] = output_id
                outputs.append(table[member_id])
            member_table_ids.append(output_id)

        return self.replace_tables(
            tables=outputs,
            member_table_ids=member_table_ids,
        )

    @property
    def num_groups(self) -> int:
        """Return the number of table groups."""
        return len(self._groups)

    def _tensors(self) -> Iterator[Tensor]:
        yield from self._groups

    def _apply_tensor(self, fn: Callable[[Tensor], Tensor]) -> Self:
        return self.replace_groups(
            [cast(TableTensor, fn(group)) for group in self._groups]
        )

    def select_stypes(
        self,
        stypes: StypeLike | Iterable[StypeLike],
    ) -> Self:
        r"""Return an ensemble table containing only ``stypes`` columns.

        Args:
            stypes: The semantic type or semantic types to select.

        Returns:
            An ensemble table preserving its logical member assignment.
        """
        if isinstance(stypes, (str, Stype)):
            stypes = (stypes,)
        stypes = tuple(Stype(stype) for stype in stypes)

        if len(stypes) == 0:
            return self.replace_groups(
                [group.select_columns(()) for group in self._groups]
            )
        return self.replace_groups(
            [group.select_stypes(stypes) for group in self._groups]
        )

    @classmethod
    def concatenate_columns(cls, tables: Sequence[Self]) -> Self:
        """Concatenate columns of corresponding ensemble members.

        Args:
            tables: Ensemble tables with equal member counts.
        """
        nonempty = tuple(table for table in tables if table.active_stypes)
        tables = nonempty or tables
        if len(tables) == 1:
            return tables[0]
        return EnsembleData.execute(torch.cat, tables, dim=-1)

    def replace_tables(
        self, tables: Sequence[TableTensor], member_table_ids: Sequence[int]
    ) -> Self:
        """Replace logical members, packing compatible replacement tables.

        Args:
            tables: Replacement tables available to the ensemble members.
            member_table_ids: Index into tables for each logical member.
        """
        if len(member_table_ids) != len(self):
            raise ValueError("Expected one replacement table per member")
        table_ids = tuple(dict.fromkeys(member_table_ids))
        groups, locations = self._pack_tables(
            tuple(tables[index] for index in table_ids)
        )
        by_table = dict(zip(table_ids, locations))
        return self.__class__(
            groups=groups,
            locations=tuple(by_table[index] for index in member_table_ids),
        )

    def replace_groups(self, groups: Sequence[TableTensor]) -> Self:
        """Return an ensemble table with its groups replaced.

        Args:
            groups: One replacement group per current group.

        Returns:
            An ensemble table over ``groups`` with the current member
            assignment.
        """
        if len(groups) != self.num_groups:
            raise ValueError(
                f"Expected one replacement per group "
                f"({self.num_groups}), got {len(groups)}."
            )
        return self.__class__(groups=groups, locations=self._locations)

    def __repr__(self, *, indent: int = 0) -> str:
        return (
            f"{' ' * indent}{self.__class__.__name__}("
            f"num_members={len(self)}, num_groups={self.num_groups})"
        )


TableLike: TypeAlias = TableTensor | EnsembleTable
