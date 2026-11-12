# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import abc
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any, Self, TypeVar, cast

import torch
from torch import Tensor

from sdm import EnsembleTable, EnsembleTensor, Stype, TableLike
from sdm.nn._buffer import _EnsembleTensorBuffer

if TYPE_CHECKING:
    from sdm.processing import Sequential

TableT = TypeVar("TableT", bound=TableLike)


class Processor(torch.nn.Module, abc.ABC):
    r"""Base processor for tensor-aware table transformations.

    A :class:`Processor` defines a reusable transformation on
    :class:`~sdm.tensor.TableTensor` or :class:`~sdm.EnsembleTable` for
    feature, target and output
    preprocessing.
    A :class:`Processor` learns any required state via :meth:`fit`, and applies
    the transformation via :meth:`transform`. Implementations preserve batch
    dimensions, and preserve rows unless documented otherwise. Batch
    dimensions are processed independently.

    :meth:`fit`, :meth:`transform`, and :meth:`fit_transform` are no-ops for
    stypes outside of :attr:`handles_stypes`.
    """

    #: Semantic types this processor operates on.
    handles_stypes: tuple[Stype, ...]

    #: Whether this processor requires fitting.
    requires_fit: bool

    _fitted_state: Tensor

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("_fitted_state", torch.tensor(False))
        self.__fitted = False

    def register_buffer(
        self,
        name: str,
        tensor: Tensor | EnsembleTensor[Any] | None,
        persistent: bool = True,
    ) -> None:
        """Register tensor or ensemble tensor state.

        Args:
            name: Attribute name of the buffer.
            tensor: Tensor, ensemble tensor, or an uninitialized buffer.
            persistent: Whether to include the buffer in ``state_dict``.
        """
        if isinstance(tensor, EnsembleTensor):
            self._buffers.pop(name, None)
            self.add_module(
                name,
                _EnsembleTensorBuffer(
                    groups=tensor._groups,
                    locations=tensor._locations,
                    persistent=persistent,
                ),
            )
            if persistent:
                self._non_persistent_buffers_set.discard(name)
            else:
                self._non_persistent_buffers_set.add(name)
            return
        if isinstance(self._modules.get(name), _EnsembleTensorBuffer):
            del self._modules[name]
        super().register_buffer(
            name, cast(Tensor | None, tensor), persistent=persistent
        )

    def __setattr__(self, name: str, value: Any) -> None:
        module = self.__dict__.get("_modules", {}).get(name)
        if isinstance(module, _EnsembleTensorBuffer) and (
            value is None or isinstance(value, (Tensor, EnsembleTensor))
        ):
            self.register_buffer(name, value, persistent=module._persistent)
            return
        if isinstance(value, EnsembleTensor) and name in self.__dict__.get(
            "_buffers", {}
        ):
            self.register_buffer(
                name,
                value,
                persistent=name not in self._non_persistent_buffers_set,
            )
            return
        super().__setattr__(name, value)

    @staticmethod
    def _merge_stypes(processors: Iterable[Processor]) -> tuple[Stype, ...]:
        return tuple(
            dict.fromkeys(
                stype
                for processor in processors
                for stype in processor.handles_stypes
            )
        )

    @property
    def _fitted(self) -> bool:
        return self.__fitted

    @_fitted.setter
    def _fitted(self, value: bool) -> None:
        self._update_fitted(value)

    def _update_fitted(self, value: bool) -> None:
        # Internal callers avoid deferring the property setter until replay.
        self.__fitted = value
        self._fitted_state.fill_(value)

    @property
    def is_fitted(self) -> bool:
        """Whether the processor has all state required for transformation."""
        return not self.requires_fit or self._fitted

    @staticmethod
    def as_processor(processor: object) -> Processor:
        r"""Normalize a processor-like object to a :class:`Processor`.

        Args:
            processor: A processor-like object. A :class:`Processor` is
                returned as-is, a callable is wrapped as a stateless processor,
                and a sequence of processor-like objects is normalized to
                :class:`~sdm.processing.common.Sequential`.
        """
        from sdm.processing import Callable, Sequential  # noqa: PLC0415

        if isinstance(processor, Processor):
            return processor
        if callable(processor):
            return Callable(processor)  # type: ignore
        if isinstance(processor, Sequence) and not isinstance(processor, str):
            return Sequential(*processor)
        raise TypeError(
            f"Input must be a 'Processor', callable, or sequence of them "
            f"(got '{type(processor).__name__}')"
        )

    def _save_to_state_dict(
        self, destination: dict[str, Any], prefix: str, keep_vars: bool
    ) -> None:
        super()._save_to_state_dict(destination, prefix, keep_vars)
        for name, buffer in self._buffers.items():
            if (
                name not in self._non_persistent_buffers_set
                and isinstance(buffer, Tensor)
                and type(buffer) is not Tensor
            ):
                destination[f"{prefix}{name}"] = (
                    buffer if keep_vars else buffer.clone()
                )

    def _load_from_state_dict(
        self,
        state_dict: dict[str, Any],
        prefix: str,
        local_metadata: dict[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        for name in tuple(self._buffers):
            key = f"{prefix}{name}._extra_state"
            if key in state_dict:
                num_groups, locations = state_dict[key]
                buffer = self._buffers[name]
                device = (
                    buffer.device
                    if buffer is not None
                    else self._fitted_state.device
                )
                self.register_buffer(
                    name,
                    EnsembleTensor(
                        groups=tuple(
                            torch.empty_like(
                                state_dict[f"{prefix}{name}.{index}"],
                                device=device,
                            )
                            if type(state_dict[f"{prefix}{name}.{index}"])
                            is Tensor
                            else state_dict[f"{prefix}{name}.{index}"].to(
                                device
                            )
                            for index in range(num_groups)
                        ),
                        locations=locations,
                    ),
                )
        for name, module in tuple(self._modules.items()):
            if isinstance(module, _EnsembleTensorBuffer) and isinstance(
                state_dict.get(f"{prefix}{name}"), Tensor
            ):
                self.register_buffer(
                    name,
                    torch.empty_like(
                        state_dict[f"{prefix}{name}"],
                        device=self._fitted_state.device,
                    ),
                )
        materialized = []
        # Resize uninitialized buffers before PyTorch copies saved values.
        for name, buffer in self._buffers.items():
            state = state_dict.get(f"{prefix}{name}")
            device = (
                buffer.device
                if buffer is not None
                else self._fitted_state.device
            )
            if isinstance(state, Tensor) and type(state) is not Tensor:
                self._buffers[name] = (
                    state_dict.pop(f"{prefix}{name}").to(device).clone()
                )
                materialized.append(f"{prefix}{name}")
                continue
            if buffer is None and isinstance(state, Tensor):
                self._buffers[name] = torch.empty_like(state, device=device)
                continue
            if (
                buffer is not None
                and isinstance(state, Tensor)
                and (
                    buffer.shape != state.shape or buffer.dtype != state.dtype
                )
            ):
                self._buffers[name] = buffer.new_empty(
                    state.shape,
                    dtype=state.dtype,
                )
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        for key in materialized:
            if key in missing_keys:
                missing_keys.remove(key)
        self.__fitted = bool(self._fitted_state)

    def _set_buffer(
        self, name: str, value: Tensor | EnsembleTensor[Any]
    ) -> None:
        # Use register_buffer to avoid assignment errors in torch.compile.
        self.register_buffer(
            name,
            value,
            persistent=name not in self._non_persistent_buffers_set,
        )

    def _set_fitted(self, device: torch.device) -> None:
        # Create fitted state explicitly so tracing preserves its value.
        self._set_buffer(
            "_fitted_state", torch.ones_like(self._fitted_state, device=device)
        )
        self._update_fitted(True)

    def _check_is_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError(
                f"{self.__class__.__name__!r} is not fitted; "
                "call 'fit()' before."
            )

    def _fit(
        self,
        table: TableLike,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def _transform(self, table: TableLike) -> TableLike:
        pass

    def _fit_transform(
        self,
        table: TableLike,
        *,
        generator: torch.Generator | None = None,
    ) -> TableLike:
        if self.requires_fit:
            self._fit(table, generator=generator)
        return self._transform(table)

    def _handles_table(self, table: TableLike) -> bool:
        if isinstance(table, EnsembleTable):
            return any(
                stype in table.active_stypes for stype in self.handles_stypes
            )
        # Check column counts directly so torch.compile can trace this without
        # constructing or intersecting sets of column types.
        for stype in self.handles_stypes:
            if getattr(table, stype.value).size(-1) > 0:
                return True
        return False

    def fit(
        self,
        table: TableLike,
        *,
        generator: torch.Generator | None = None,
    ) -> Self:
        r"""Fit the processor.

        Args:
            table: The table used to compute the processor state.
            generator: Pseudorandom number generator used for sampling.
        """
        if not self._handles_table(table):
            return self
        if self.requires_fit:
            self._fit(table, generator=generator)
            self._set_fitted(table.device)
        return self

    def transform(self, table: TableT) -> TableT:
        r"""Transform ``table``.

        Args:
            table: The table to transform.

        Returns:
            The transformed table.
        """
        if not self._handles_table(table):
            return table
        self._check_is_fitted()
        return cast(TableT, self._transform(table))

    def forward(self, table: TableT) -> TableT:
        r"""Alias of :meth:`transform`."""
        return self.transform(table)

    def fit_transform(
        self,
        table: TableT,
        *,
        generator: torch.Generator | None = None,
    ) -> TableT:
        r"""Fit the processor and transform ``table``.

        Args:
            table: The table to fit on and transform.
            generator: Pseudorandom number generator used for sampling.

        Returns:
            The transformed table.
        """
        if not self._handles_table(table):
            return table
        out = self._fit_transform(table, generator=generator)
        if self.requires_fit:
            self._set_fitted(table.device)
        return cast(TableT, out)

    def __add__(self, other: object) -> Sequential:
        from sdm.processing import Sequential  # noqa: PLC0415

        try:
            return Sequential(self, other)
        except TypeError:
            return NotImplemented

    def __radd__(self, other: object) -> Sequential:
        from sdm.processing import Sequential  # noqa: PLC0415

        try:
            return Sequential(other, self)
        except TypeError:
            return NotImplemented

    def __repr__(self, *, indent: int = 0) -> str:
        return f"{' ' * indent}{self.__class__.__name__}()"


class InvertibleMixin(abc.ABC):
    r"""Extend a :class:`Processor` by an inverse transformation."""

    @abc.abstractmethod
    def _inverse_transform(self, table: TableLike) -> TableLike: ...

    def inverse_transform(self, table: TableT) -> TableT:
        r"""Apply the inverse transformation to ``table``.

        Args:
            table: The table in transformed representation.

        Returns:
            The table restored to the representation before
            :meth:`~Processor.transform`.
        """
        if not self._handles_table(table):
            return table
        self._check_is_fitted()
        return cast(TableT, self._inverse_transform(table))

    if TYPE_CHECKING:
        # Provided at runtime by `Processor` via the MRO.
        def _check_is_fitted(self) -> None: ...

        def _handles_table(self, table: TableLike) -> bool: ...
