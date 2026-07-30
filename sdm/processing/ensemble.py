from __future__ import annotations

import copy
import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import cast

import torch
from typing_extensions import Self

from sdm.processing.base import InvertibleMixin, Processor
from sdm.relational import RelatedTables, Relationship, TaskLink
from sdm.stype import Stype
from sdm.tensor import (
    CategoricalTensor,
    ColumnarTensor,
    StringTensor,
    TableTensor,
)


def _variant_metadata_key(table: TableTensor) -> tuple[object, ...]:
    categorical = tuple(
        (
            id(category),
            tuple(category.size()),
            category.dtype,
            category.device,
        )
        for category in table.categorical.categories
    )
    return (
        tuple(table.size()),
        tuple((stype, columns) for stype, columns in table.columns.items()),
        tuple(
            (stype, type(block), block.dtype) for stype, block in table.items()
        ),
        table.device,
        categorical,
    )


def _stack_physical(tables: Sequence[TableTensor]) -> TableTensor:
    if len(tables) == 0:
        raise ValueError("Expected at least one table to materialize.")
    if len(tables) == 1:
        return cast(TableTensor, tables[0].unsqueeze(0))

    reference = tables[0]
    reference_size = tuple(reference.size()[:-1])
    reference_widths = {
        stype: block.size(-1) for stype, block in reference.items()
    }
    for table in tables[1:]:
        if tuple(table.size()[:-1]) != reference_size:
            raise ValueError(
                "Cannot materialize ensemble members with different row or "
                "batch dimensions."
            )
        widths = {stype: block.size(-1) for stype, block in table.items()}
        if widths != reference_widths:
            raise ValueError(
                "Cannot materialize ensemble members with incompatible "
                "semantic-type widths."
            )
        for (stype, actual), (_, expected) in zip(
            table.items(),
            reference.items(),
        ):
            if actual.dtype != expected.dtype:
                raise ValueError(
                    "Cannot materialize ensemble members with different "
                    f"{stype.value!r} dtypes."
                )
            if actual.device != expected.device:
                raise ValueError(
                    "Cannot materialize ensemble members on different devices."
                )

    blocks: dict[Stype, torch.Tensor] = {}
    for stype, reference_block in reference.items():
        member_blocks = [table.blocks[stype] for table in tables]
        if stype == Stype.categorical:
            categories = reference.categorical.categories
            expected_counts = tuple(
                category.numel() for category in categories
            )
            for table in tables[1:]:
                if (
                    tuple(
                        category.numel()
                        for category in table.categorical.categories
                    )
                    != expected_counts
                ):
                    raise ValueError(
                        "Cannot materialize categorical ensemble members "
                        "with different class counts."
                    )
            code = torch.stack(
                [table.categorical.code for table in tables],
                dim=0,
            )
            blocks[stype] = CategoricalTensor(
                code=code,
                categories=categories,
            )
        elif reference_block.size(-1) > 0:
            blocks[stype] = torch.stack(member_blocks, dim=0)

    return TableTensor(
        size=(len(tables), *reference_size),
        columns={
            stype.value: columns
            for stype, columns in reference.columns.items()
        },
        device=reference.device,
        numerical=blocks.get(Stype.numerical),
        categorical=cast(
            CategoricalTensor | None,
            blocks.get(Stype.categorical),
        ),
        datetime=blocks.get(Stype.datetime),
        text=cast(StringTensor | None, blocks.get(Stype.text)),
        id=cast(ColumnarTensor | None, blocks.get(Stype.id)),
    )


@dataclass(frozen=True)
class EnsembleTable:
    r"""Store shared and distinct table variants for ensemble members.

    Every group has shape ``[V, ..., R, C]``. The stable member mapping points
    to a group and variant position without inferring equality from tensor
    values.

    Args:
        groups: Compatible table variants grouped along their leading
            dimension.
        member_to_variant: ``(group, variant)`` location for every member.
    """

    groups: tuple[TableTensor, ...]
    member_to_variant: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        if len(self.groups) == 0:
            raise ValueError("Expected at least one ensemble group.")
        if len(self.member_to_variant) == 0:
            raise ValueError("Expected at least one ensemble member.")

        devices = {group.device for group in self.groups}
        if len(devices) != 1:
            raise ValueError(
                "Expected all ensemble groups to use the same device."
            )

        for group_index, group in enumerate(self.groups):
            if group.dim() < 3:
                raise ValueError(
                    "Expected ensemble groups with shape "
                    f"[V, ..., R, C] (group {group_index} is {group.dim()}D)."
                )
            if group.size(0) == 0:
                raise ValueError(
                    "Expected every ensemble group to be non-empty."
                )

        for member, (group, variant) in enumerate(self.member_to_variant):
            if not 0 <= group < len(self.groups):
                raise ValueError(
                    f"Member {member} references unknown group {group}."
                )
            if not 0 <= variant < self.groups[group].size(0):
                raise ValueError(
                    f"Member {member} references unknown variant {variant} "
                    f"in group {group}."
                )

    @classmethod
    def from_shared(
        cls,
        table: TableTensor,
        *,
        num_members: int,
    ) -> Self:
        r"""Create an ensemble whose members share one table.

        Args:
            table: Shared table with shape ``[..., R, C]``.
            num_members: Positive number of logical ensemble members.
        """
        if num_members < 1:
            raise ValueError("'num_members' needs to be positive.")
        group = cast(TableTensor, table.unsqueeze(0))
        return cls(
            groups=(group,),
            member_to_variant=((0, 0),) * num_members,
        )

    @classmethod
    def pack(
        cls,
        variants: Sequence[TableTensor],
        member_to_input_variant: Sequence[int],
    ) -> Self:
        r"""Pack proven variants into compatible physical groups.

        Args:
            variants: Distinct results in provenance order.
            member_to_input_variant: Input variant index for every member.
        """
        variants = tuple(variants)
        member_to_input_variant = tuple(member_to_input_variant)
        if len(variants) == 0:
            raise ValueError("Expected at least one input variant.")
        if len(member_to_input_variant) == 0:
            raise ValueError("Expected at least one ensemble member.")
        if any(
            variant < 0 or variant >= len(variants)
            for variant in member_to_input_variant
        ):
            raise ValueError(
                "'member_to_input_variant' references an unknown variant."
            )

        buckets: dict[tuple[object, ...], list[int]] = {}
        for index, table in enumerate(variants):
            buckets.setdefault(_variant_metadata_key(table), []).append(index)

        groups: list[TableTensor] = []
        input_to_location: dict[int, tuple[int, int]] = {}
        for indices in buckets.values():
            group_index = len(groups)
            group = (
                cast(TableTensor, variants[indices[0]].unsqueeze(0))
                if len(indices) == 1
                else cast(
                    TableTensor,
                    torch.stack(
                        [variants[index] for index in indices],
                        dim=0,
                    ),
                )
            )
            groups.append(group)
            input_to_location.update(
                {
                    input_index: (group_index, variant_index)
                    for variant_index, input_index in enumerate(indices)
                }
            )

        return cls(
            groups=tuple(groups),
            member_to_variant=tuple(
                input_to_location[index] for index in member_to_input_variant
            ),
        )

    @property
    def num_members(self) -> int:
        """Return the number of logical ensemble members."""
        return len(self.member_to_variant)

    @property
    def device(self) -> torch.device:
        """Return the common device of all groups."""
        return self.groups[0].device

    def __getitem__(self, member_id: int) -> TableTensor:
        group, variant = self.member_to_variant[member_id]
        return self.groups[group][variant]

    def iter_variants(
        self,
    ) -> tuple[tuple[tuple[int, int], TableTensor], ...]:
        r"""Return stored variants in group order."""
        return tuple(
            (
                (group_index, variant_index),
                group[variant_index],
            )
            for group_index, group in enumerate(self.groups)
            for variant_index in range(group.size(0))
        )

    def map_variants(
        self,
        function: Callable[[TableTensor], TableTensor],
    ) -> Self:
        r"""Apply ``function`` once per stored variant.

        Args:
            function: Row-preserving table transformation.
        """
        entries = self.iter_variants()
        variants = tuple(function(table) for _, table in entries)
        location_to_input = {
            location: index for index, (location, _) in enumerate(entries)
        }
        return self.pack(
            variants=variants,
            member_to_input_variant=tuple(
                location_to_input[location]
                for location in self.member_to_variant
            ),
        )

    def select_members(self, member_ids: Sequence[int]) -> Self:
        r"""Select members while preserving their requested order.

        Args:
            member_ids: Member positions to select.
        """
        member_ids = tuple(member_ids)
        locations: dict[tuple[int, int], int] = {}
        variants: list[TableTensor] = []
        member_to_input: list[int] = []
        for member_id in member_ids:
            location = self.member_to_variant[member_id]
            if location not in locations:
                locations[location] = len(variants)
                group, variant = location
                variants.append(self.groups[group][variant])
            member_to_input.append(locations[location])
        return self.pack(
            variants=variants,
            member_to_input_variant=member_to_input,
        )

    def materialize(
        self,
        member_ids: Sequence[int] | None = None,
    ) -> TableTensor:
        r"""Materialize members without aligning permuted columns by name.

        Args:
            member_ids: Optional member positions. All members are used by
                default.
        """
        if member_ids is None:
            member_ids = tuple(range(self.num_members))
        tables = tuple(self[member_id] for member_id in member_ids)
        return _stack_physical(tables)

    def with_groups(self, groups: tuple[TableTensor, ...]) -> Self:
        r"""Replace physical groups while retaining the member mapping.

        Args:
            groups: Replacement groups with unchanged variant counts.
        """
        if len(groups) != len(self.groups) or any(
            actual.size(0) != expected.size(0)
            for actual, expected in zip(groups, self.groups)
        ):
            raise ValueError(
                "Expected replacement groups to preserve group and variant "
                "counts."
            )
        return self.__class__(
            groups=groups,
            member_to_variant=self.member_to_variant,
        )


@dataclass(frozen=True)
class EnsembleRelatedTables:
    r"""Store ensemble variants for every logical related table.

    Args:
        tables: Ensemble tables keyed by logical table name.
        relationships: Shared relationships among the tables.
        task_links: Shared links from task rows to related tables.
    """

    tables: Mapping[str, EnsembleTable]
    relationships: tuple[Relationship, ...]
    task_links: tuple[TaskLink, ...]

    def __post_init__(self) -> None:
        member_counts = {table.num_members for table in self.tables.values()}
        if len(member_counts) > 1:
            raise ValueError(
                "Expected every related ensemble table to have the same "
                "number of members."
            )

    def __getitem__(self, table_name: str) -> EnsembleTable:
        return self.tables[table_name]

    def member(self, member_id: int) -> RelatedTables:
        r"""Return one member's related tables.

        Args:
            member_id: Stable member position.
        """
        return RelatedTables(
            tables={
                name: table[member_id] for name, table in self.tables.items()
            },
            relationships=self.relationships,
            task_links=self.task_links,
        )

    def materialize(self, member_ids: Sequence[int]) -> RelatedTables:
        r"""Materialize selected members for model execution.

        Args:
            member_ids: Stable member positions.
        """
        return RelatedTables(
            tables={
                name: table.materialize(member_ids)
                for name, table in self.tables.items()
            },
            relationships=self.relationships,
            task_links=self.task_links,
        )


class EnsemblePlanner:
    r"""Optionally coordinate member decisions across Processor paths.

    A planner is initialized once per Recipe fit. Structural Processors may
    request explicit member mappings from it; returning ``None`` keeps their
    normal member-local RNG semantics.
    """

    def initialize(
        self,
        *,
        features: TableTensor,
        target: TableTensor,
        num_members: int,
        seed: int,
    ) -> None:
        r"""Initialize a new fit plan.

        Args:
            features: Raw context feature table.
            target: Raw context target table.
            num_members: Number of logical ensemble members.
            seed: Root fit seed.
        """
        del features, target, num_members, seed

    def column_permutations(
        self,
        *,
        member_ids: tuple[int, ...],
        num_columns: tuple[int, ...],
        table_scope: str,
        processor_path: tuple[str, ...],
    ) -> tuple[tuple[int, ...], ...] | None:
        r"""Return optional explicit column permutations.

        Args:
            member_ids: Stable global member ids.
            num_columns: Column count for each member input.
            table_scope: Logical table fit scope.
            processor_path: Stable path of the requesting Processor.
        """
        del member_ids, num_columns, table_scope, processor_path
        return None

    def category_permutations(
        self,
        *,
        member_ids: tuple[int, ...],
        category_counts: tuple[tuple[int, ...], ...],
        table_scope: str,
        processor_path: tuple[str, ...],
    ) -> tuple[tuple[tuple[int, ...], ...], ...] | None:
        r"""Return optional explicit per-column category permutations.

        Args:
            member_ids: Stable global member ids.
            category_counts: Category counts by member and column.
            table_scope: Logical table fit scope.
            processor_path: Stable path of the requesting Processor.
        """
        del member_ids, category_counts, table_scope, processor_path
        return None

    def canonical_classes(self) -> tuple[object, ...] | None:
        r"""Return an optional canonical observed classification order."""
        return None


@dataclass(frozen=True)
class EnsembleFitContext:
    r"""Describe stable member identity and fit scope during execution.

    Args:
        member_ids: Global member ids aligned with the current input.
        base_seed: Root seed for member-local random streams.
        table_scope: Logical table fit scope.
        processor_path: Stable path of the current processor.
        planner: Optional cross-path member-decision planner.
    """

    member_ids: tuple[int, ...]
    base_seed: int
    table_scope: str
    processor_path: tuple[str, ...] = ()
    planner: EnsemblePlanner | None = None

    @classmethod
    def create(
        cls,
        *,
        num_members: int,
        table_scope: str,
        generator: torch.Generator | None = None,
        base_seed: int | None = None,
        planner: EnsemblePlanner | None = None,
    ) -> Self:
        r"""Create a root fit context.

        Args:
            num_members: Positive number of ensemble members.
            table_scope: Logical table fit scope.
            generator: Optional user-controlled root generator.
            base_seed: Optional precomputed root seed.
            planner: Optional shared member-decision planner.
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
            planner=planner,
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
            planner=self.planner,
        )

    def select_members(self, positions: Sequence[int]) -> Self:
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
            planner=self.planner,
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

    def _fit_transform(
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> TableTensor:
        output = self.fit_transform_ensemble(
            EnsembleTable.from_shared(table, num_members=1),
            context=self._direct_context(generator),
        )
        return output[0]

    def _transform(self, table: TableTensor) -> TableTensor:
        ensemble = EnsembleTable.from_shared(table, num_members=1)
        if not self.requires_fit:
            return self.fit_transform_ensemble(
                ensemble,
                context=self._direct_context(None),
            )[0]
        return self.transform_ensemble(ensemble)[0]

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

    def inverse_transform_members(
        self,
        tables: Sequence[TableTensor],
    ) -> tuple[TableTensor, ...]:
        r"""Inverse-transform member-aligned tables.

        Args:
            tables: One transformed table per stable member.
        """
        raise AttributeError(
            f"{self.__class__.__name__!r} object has no attribute "
            "'inverse_transform_members'"
        )


class _EnsembleProcessorAdapter(EnsembleProcessor):
    def __init__(self, processor: Processor) -> None:
        super().__init__()
        self.template = processor
        self.requires_fit = processor.requires_fit
        self.processors = torch.nn.ModuleList()
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

        groups = tuple(
            cast(Processor, processor).transform(group)
            for group, processor in zip(table.groups, self.processors)
        )
        return table.with_groups(groups)

    def inverse_transform_members(
        self,
        tables: Sequence[TableTensor],
    ) -> tuple[TableTensor, ...]:
        tables = tuple(tables)
        if self._variable_schema:
            raise TypeError("Variable-schema processors are not invertible.")
        if len(self.processors) != 1:
            raise NotImplementedError(
                "Member inverse transform across multiple fitted groups "
                "requires a structural EnsembleProcessor."
            )
        processor = self.processors[0]
        if not isinstance(processor, InvertibleMixin):
            raise TypeError(
                f"{processor.__class__.__name__!r} is not invertible."
            )
        restored = processor.inverse_transform(_stack_physical(tables))
        return tuple(
            cast(TableTensor, restored[index])
            for index in range(restored.size(0))
        )


def as_ensemble_processor(processor: Processor) -> EnsembleProcessor:
    r"""Return an ensemble Processor, adapting a normal leaf if needed.

    Args:
        processor: Processor to normalize.
    """
    if isinstance(processor, EnsembleProcessor):
        return processor
    return _EnsembleProcessorAdapter(processor)
