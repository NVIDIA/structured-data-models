from __future__ import annotations

import copy
import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, cast

import torch
from typing_extensions import Self

from sdm.processing.base import InvertibleMixin, Processor
from sdm.processing.ensemble_table import EnsembleTable, _stack_positional
from sdm.tensor import TableTensor


class _EnsemblePlan(Protocol):
    def initialize(
        self,
        *,
        target: TableTensor,
        num_members: int,
        seed: int,
    ) -> None: ...

    def column_permutations(
        self,
        *,
        member_ids: tuple[int, ...],
        num_columns: tuple[int, ...],
        table_scope: str,
    ) -> tuple[tuple[int, ...], ...] | None: ...

    def category_permutations(
        self,
        *,
        member_ids: tuple[int, ...],
        category_counts: tuple[tuple[int, ...], ...],
        table_scope: str,
    ) -> tuple[tuple[tuple[int, ...], ...], ...] | None: ...

    def canonical_classes(self) -> tuple[object, ...] | None: ...


@dataclass(frozen=True)
class EnsembleFitContext:
    r"""Describe stable member identity and fit scope during execution.

    Args:
        member_ids: Global member ids aligned with the current input.
        base_seed: Root seed for member-local random streams.
        table_scope: Logical table fit scope.
        processor_path: Stable path of the current processor.
    """

    member_ids: tuple[int, ...]
    base_seed: int
    table_scope: str
    processor_path: tuple[str, ...] = ()
    _plan: _EnsemblePlan | None = None

    @classmethod
    def create(
        cls,
        *,
        num_members: int,
        table_scope: str,
        generator: torch.Generator | None = None,
        base_seed: int | None = None,
        _plan: _EnsemblePlan | None = None,
    ) -> Self:
        r"""Create a root fit context.

        Args:
            num_members: Positive number of ensemble members.
            table_scope: Logical table fit scope.
            generator: Optional user-controlled root generator.
            base_seed: Optional precomputed root seed.
        """
        if num_members < 1:
            raise ValueError("'num_members' needs to be positive.")
        if base_seed is not None and generator is not None:
            raise ValueError("Provide either 'generator' or 'base_seed'.")
        if base_seed is None and generator is None:
            base_seed = int(
                torch.empty((), dtype=torch.int64).random_().item()
            )
        elif base_seed is None:
            assert generator is not None
            base_seed = generator.initial_seed()
        return cls(
            member_ids=tuple(range(num_members)),
            base_seed=base_seed,
            table_scope=table_scope,
            _plan=_plan,
        )

    def child(self, name: str) -> Self:
        r"""Return a context for a child processor.

        Args:
            name: Stable child path segment.
        """
        return self.__class__(
            member_ids=self.member_ids,
            base_seed=self.base_seed,
            table_scope=self.table_scope,
            processor_path=(*self.processor_path, name),
            _plan=self._plan,
        )

    def _select_members(self, positions: Sequence[int]) -> Self:
        r"""Select local member positions.

        Args:
            positions: Positions relative to the current input.
        """
        return self.__class__(
            member_ids=tuple(
                self.member_ids[position] for position in positions
            ),
            base_seed=self.base_seed,
            table_scope=self.table_scope,
            processor_path=self.processor_path,
            _plan=self._plan,
        )

    def generator_for(
        self,
        member_id: int,
        *,
        device: torch.device | str = "cpu",
    ) -> torch.Generator:
        r"""Create a stable generator for one member and processor path.

        Args:
            member_id: Global member id.
            device: Generator device.
        """
        payload = repr(
            (
                self.base_seed,
                member_id,
                self.table_scope,
                self.processor_path,
            )
        ).encode()
        seed = int.from_bytes(
            hashlib.blake2b(payload, digest_size=8).digest(),
            byteorder="little",
        )
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        return generator


class VariableSchemaBatchMixin:
    r"""Define a batched path whose variants may produce different schemas."""

    def fit_transform_batch(
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> tuple[TableTensor, ...]:
        r"""Fit and transform variants from ``table`` independently.

        Args:
            table: Variant batch with shape ``[V, ..., R, C]``.
            generator: Optional pseudorandom number generator.
        """
        self.fit_batch(table, generator=generator)
        return self.transform_batch(table)

    def fit_batch(
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> Self:
        r"""Fit a variable-schema variant batch.

        Args:
            table: Variant batch with shape ``[V, ..., R, C]``.
            generator: Optional pseudorandom number generator.
        """
        raise NotImplementedError

    def transform_batch(
        self,
        table: TableTensor,
    ) -> tuple[TableTensor, ...]:
        r"""Transform a variant batch into one table per variant.

        Args:
            table: Variant batch with shape ``[V, ..., R, C]``.
        """
        raise NotImplementedError


class EnsembleProcessor(Processor):
    r"""Base processor for provenance-aware ensemble execution."""

    @staticmethod
    def _direct_context(
        generator: torch.Generator | None,
    ) -> EnsembleFitContext:
        return EnsembleFitContext.create(
            num_members=1,
            table_scope="direct",
            generator=generator,
        )

    def _fit(
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        self.fit_transform_ensemble(
            EnsembleTable.from_shared(table, num_members=1),
            context=self._direct_context(generator),
        )

    def _transform(self, table: TableTensor) -> TableTensor:
        ensemble = EnsembleTable.from_shared(table, num_members=1)
        if not self.requires_fit:
            return self.fit_transform_ensemble(
                ensemble,
                context=self._direct_context(None),
            )[0]
        return self.transform_ensemble(ensemble)[0]

    def _fit_transform(
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> TableTensor:
        if (
            type(self)._fit is not EnsembleProcessor._fit
            or type(self)._transform is not EnsembleProcessor._transform
        ):
            return super()._fit_transform(table, generator=generator)
        return self.fit_transform_ensemble(
            EnsembleTable.from_shared(table, num_members=1),
            context=self._direct_context(generator),
        )[0]

    def fit_ensemble(
        self,
        table: EnsembleTable,
        *,
        context: EnsembleFitContext,
    ) -> Self:
        r"""Fit an ensemble table.

        Args:
            table: Ensemble fit input.
            context: Stable fit scope and member identity.
        """
        self.fit_transform_ensemble(table, context=context)
        if self.requires_fit:
            self._fitted = True
        return self

    def fit_transform_ensemble(
        self,
        table: EnsembleTable,
        *,
        context: EnsembleFitContext,
    ) -> EnsembleTable:
        r"""Fit and transform an ensemble table.

        Args:
            table: Ensemble fit input.
            context: Stable fit scope and member identity.
        """
        raise NotImplementedError

    def transform_ensemble(self, table: EnsembleTable) -> EnsembleTable:
        r"""Transform an ensemble table with fitted state.

        Args:
            table: Ensemble transform input.
        """
        raise NotImplementedError

    def inverse_transform_ensemble(
        self,
        table: EnsembleTable,
    ) -> EnsembleTable:
        r"""Inverse-transform an ensemble table.

        Args:
            table: Member-aligned transformed table.
        """
        raise AttributeError(
            f"{self.__class__.__name__!r} object has no attribute "
            "'inverse_transform_ensemble'"
        )


class _EnsembleProcessorAdapter(EnsembleProcessor):
    def __init__(self, processor: Processor) -> None:
        super().__init__()
        self.template = processor
        self.requires_fit = processor.requires_fit
        self.processors = torch.nn.ModuleList()
        self._member_to_fitted_variant: tuple[tuple[int, int], ...] = ()
        self._fitted_group_sizes: tuple[int, ...] = ()
        self._variable_schema = isinstance(
            processor,
            VariableSchemaBatchMixin,
        )

    def _validate_row_count(
        self,
        before: TableTensor,
        after: TableTensor,
    ) -> None:
        if before.size(-2) != after.size(-2):
            raise ValueError(
                f"{self.template.__class__.__name__!r} changes the row "
                "dimension and is not supported in ensemble Recipes."
            )

    def fit_transform_ensemble(
        self,
        table: EnsembleTable,
        *,
        context: EnsembleFitContext,
    ) -> EnsembleTable:
        del context
        self.processors = torch.nn.ModuleList()
        self._member_to_fitted_variant = table.member_to_variant
        self._fitted_group_sizes = tuple(
            group.size(0) for group in table.groups
        )

        if self._variable_schema:
            return self._fit_transform_variable_schema(table)

        groups: list[TableTensor] = []
        for group in table.groups:
            processor = copy.deepcopy(self.template)
            after = processor.fit_transform(group)
            self._validate_row_count(group, after)
            self.processors.append(processor)
            groups.append(after)
        return table.with_groups(tuple(groups))

    def _fit_transform_variable_schema(
        self,
        table: EnsembleTable,
    ) -> EnsembleTable:
        variants: list[TableTensor] = []
        offsets: list[int] = []
        offset = 0
        for group in table.groups:
            processor = copy.deepcopy(self.template)
            assert isinstance(processor, VariableSchemaBatchMixin)
            outputs = processor.fit_transform_batch(group)
            if len(outputs) != group.size(0):
                raise RuntimeError(
                    f"{processor.__class__.__name__!r}.fit_transform_batch() "
                    "must return one table per input variant."
                )
            for index, after in enumerate(outputs):
                before = group[index]
                self._validate_row_count(before, after)
            offsets.append(offset)
            offset += len(outputs)
            variants.extend(outputs)
            self.processors.append(cast(Processor, processor))

        member_to_input = tuple(
            offsets[group] + variant
            for group, variant in table.member_to_variant
        )
        return EnsembleTable.pack(
            variants=variants,
            member_to_input_variant=member_to_input,
        )

    def transform_ensemble(self, table: EnsembleTable) -> EnsembleTable:
        if len(table.groups) != len(self.processors):
            raise ValueError(
                "Expected transform input to preserve fitted ensemble "
                "group provenance."
            )
        if self._variable_schema:
            variants: list[TableTensor] = []
            offsets: list[int] = []
            offset = 0
            for group, processor in zip(table.groups, self.processors):
                assert isinstance(processor, VariableSchemaBatchMixin)
                outputs = processor.transform_batch(group)
                if len(outputs) != group.size(0):
                    raise RuntimeError(
                        f"{processor.__class__.__name__!r}."
                        "transform_batch() must return one table per input "
                        "variant."
                    )
                for index, after in enumerate(outputs):
                    self._validate_row_count(group[index], after)
                offsets.append(offset)
                offset += len(outputs)
                variants.extend(outputs)
            return EnsembleTable.pack(
                variants=variants,
                member_to_input_variant=tuple(
                    offsets[group] + variant
                    for group, variant in table.member_to_variant
                ),
            )

        groups: list[TableTensor] = []
        for group, processor in zip(table.groups, self.processors):
            after = cast(Processor, processor).transform(group)
            self._validate_row_count(group, after)
            groups.append(after)
        return table.with_groups(tuple(groups))

    def inverse_transform_ensemble(
        self,
        table: EnsembleTable,
    ) -> EnsembleTable:
        if self._variable_schema:
            raise TypeError("Variable-schema processors are not invertible.")

        if table.num_members != len(self._member_to_fitted_variant):
            raise ValueError(
                "Expected one inverse-transform input per fitted member."
            )
        if self._fitted_group_sizes == (1,):
            processor = self.processors[0]
            if not isinstance(processor, InvertibleMixin):
                raise TypeError(
                    f"{processor.__class__.__name__!r} is not invertible."
                )
            groups = []
            for group in table.groups:
                restored = processor.inverse_transform(group)
                self._validate_row_count(group, restored)
                groups.append(restored)
            return table.with_groups(tuple(groups))

        members_by_variant = [
            [[] for _ in range(group_size)]
            for group_size in self._fitted_group_sizes
        ]
        for member, (group, variant) in enumerate(
            self._member_to_fitted_variant
        ):
            members_by_variant[group][variant].append(member)

        variants: list[TableTensor] = []
        member_to_variant = [0] * table.num_members
        for processor, group_members in zip(
            self.processors,
            members_by_variant,
        ):
            if not isinstance(processor, InvertibleMixin):
                raise TypeError(
                    f"{processor.__class__.__name__!r} is not invertible."
                )
            rounds = max(len(members) for members in group_members)
            round_tables = []
            for round_index in range(rounds):
                # ``[round, fitted variant, ..., row, column]`` lets the
                # vectorized fitted state broadcast over repeated members.
                round_tables.append(
                    _stack_positional(
                        tuple(
                            table[members[min(round_index, len(members) - 1)]]
                            for members in group_members
                        )
                    )
                )
            batch = _stack_positional(round_tables)
            restored = processor.inverse_transform(batch)
            self._validate_row_count(batch, restored)
            for fitted_variant, members in enumerate(group_members):
                for round_index, member in enumerate(members):
                    member_to_variant[member] = len(variants)
                    variants.append(restored[round_index, fitted_variant])

        return EnsembleTable.pack(
            variants=variants,
            member_to_input_variant=member_to_variant,
        )


def as_ensemble_processor(processor: Processor) -> EnsembleProcessor:
    r"""Return an ensemble Processor, adapting a normal leaf if needed.

    Args:
        processor: Processor to normalize.
    """
    if isinstance(processor, EnsembleProcessor):
        return processor
    return _EnsembleProcessorAdapter(processor)
