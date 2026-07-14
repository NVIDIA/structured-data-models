from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import Tensor
from typing_extensions import Self

from sdm.tensor.mixin import DeviceMixin

_INTEGER_DTYPES = {
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
}
_INDEX_DTYPES = {torch.int32, torch.int64}


@dataclass(frozen=True, repr=False)
class RelationalSample(DeviceMixin):
    r"""Exact neighborhood sample attached to task-specific related tables.

    Relationship and task edge tensors are aligned positionally with the
    relationships and task links in the associated
    :class:`~sdm.RelatedTables`. Relationship edges use their canonical
    left-to-right orientation. Sampling an edge in either traversal direction
    includes it once, at its earliest discovery hop.

    Args:
        node_batch: Example index for every sampled table row.
        node_hops: Discovery hop for every sampled table row. Seeds are at hop
            zero.
        edge_indices: Exact sampled edges, one tensor per relationship, each
            with shape ``[2, num_edges]``.
        edge_hops: Discovery hop for every edge in ``edge_indices``.
        task_edge_indices: Exact task-to-seed edges, one tensor per task link.
            Source indices address task rows and destination indices address
            rows in the linked related table.
        num_hops: Configured neighborhood sampling depth. This preserves
            trailing empty hops that cannot be inferred from node tensors.
        num_neighbors: Configured fanout at each sampling hop. A value of
            ``-1`` samples every neighbor.
        disjoint: Whether each task row was sampled as a disjoint graph.
        temporal: Whether temporal node constraints were active.
        temporal_strategy: Temporal neighbor selection strategy.
        seed_time: Optional anchor time for every task row, expressed in the
            same units as :class:`~sdm.TableTensor` datetime columns.
    """

    node_batch: Mapping[str, Tensor]
    node_hops: Mapping[str, Tensor]
    edge_indices: tuple[Tensor, ...]
    edge_hops: tuple[Tensor, ...]
    task_edge_indices: tuple[Tensor, ...]
    num_hops: int
    num_neighbors: tuple[int, ...]
    disjoint: bool
    temporal: bool
    temporal_strategy: str
    seed_time: Tensor | None = None

    def __post_init__(self) -> None:
        if self.num_hops != len(self.num_neighbors):
            raise ValueError(
                "Expected 'num_hops' to equal the configured fanout depth"
            )
        if self.node_batch.keys() != self.node_hops.keys():
            raise ValueError(
                "Expected 'node_batch' and 'node_hops' to have the same "
                "table names"
            )
        if len(self.edge_indices) != len(self.edge_hops):
            raise ValueError(
                "Expected 'edge_indices' and 'edge_hops' to have the same "
                "length"
            )

        for table_name, batch in self.node_batch.items():
            hop = self.node_hops[table_name]
            _validate_index_vector(batch, f"batch for table {table_name!r}")
            _validate_integer_vector(
                hop, f"node hops for table {table_name!r}"
            )
            if batch.size() != hop.size():
                raise ValueError(
                    f"Expected batch and node hops for table {table_name!r} "
                    "to have the same size"
                )
        for index, (edge_index, edge_hop) in enumerate(
            zip(self.edge_indices, self.edge_hops)
        ):
            _validate_edge_index(edge_index, f"edge index {index}")
            _validate_integer_vector(edge_hop, f"edge hops at index {index}")
            if edge_hop.numel() != edge_index.size(1):
                raise ValueError(
                    f"Expected edge index and hops at index {index} to have "
                    "the same number of edges"
                )
        for index, edge_index in enumerate(self.task_edge_indices):
            _validate_edge_index(edge_index, f"task edge index {index}")

        if self.seed_time is not None:
            if self.seed_time.dim() != 1:
                raise ValueError("Expected 'seed_time' to be one-dimensional")
            if self.seed_time.dtype != torch.int64:
                raise ValueError("Expected 'seed_time' to have dtype int64")

    def to(self, device: torch.device | str | None) -> Self:
        r""":meta private:"""  # noqa: D415
        if device is None or self.device == torch.device(device):
            return self
        return self.__class__(
            node_batch={
                table_name: batch.to(device)
                for table_name, batch in self.node_batch.items()
            },
            node_hops={
                table_name: hop.to(device)
                for table_name, hop in self.node_hops.items()
            },
            edge_indices=tuple(
                edge_index.to(device) for edge_index in self.edge_indices
            ),
            edge_hops=tuple(
                edge_hop.to(device) for edge_hop in self.edge_hops
            ),
            task_edge_indices=tuple(
                edge_index.to(device) for edge_index in self.task_edge_indices
            ),
            num_hops=self.num_hops,
            num_neighbors=self.num_neighbors,
            disjoint=self.disjoint,
            temporal=self.temporal,
            temporal_strategy=self.temporal_strategy,
            seed_time=(
                None if self.seed_time is None else self.seed_time.to(device)
            ),
        )

    @property
    def device(self) -> torch.device:
        r""":meta private:"""  # noqa: D415
        tensors = (
            *self.node_batch.values(),
            *self.node_hops.values(),
            *self.edge_indices,
            *self.edge_hops,
            *self.task_edge_indices,
        )
        if self.seed_time is not None:
            tensors = (*tensors, self.seed_time)
        devices = {tensor.device for tensor in tensors}
        if len(devices) == 0:
            raise RuntimeError(
                f"Could not determine 'device' of empty "
                f"'{self.__class__.__name__}'"
            )
        if len(devices) > 1:
            raise RuntimeError(
                f"Expected relational sample tensors to be on the same "
                f"device (got {list(devices)})"
            )
        return next(iter(devices))

    def __repr__(self) -> str:
        nodes = {
            name: batch.numel() for name, batch in self.node_batch.items()
        }
        edges = tuple(edge_index.size(1) for edge_index in self.edge_indices)
        return (
            f"{self.__class__.__name__}(num_hops={self.num_hops}, "
            f"nodes={nodes}, edges={edges})"
        )


def _validate_edge_index(edge_index: Tensor, name: str) -> None:
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError(f"Expected {name} to have shape [2, num_edges]")
    if edge_index.dtype not in _INDEX_DTYPES:
        raise ValueError(f"Expected {name} to have dtype int32 or int64")


def _validate_integer_vector(tensor: Tensor, name: str) -> None:
    if tensor.dim() != 1:
        raise ValueError(f"Expected {name} to be one-dimensional")
    if tensor.dtype not in _INTEGER_DTYPES:
        raise ValueError(f"Expected {name} to be integer")


def _validate_index_vector(tensor: Tensor, name: str) -> None:
    if tensor.dim() != 1:
        raise ValueError(f"Expected {name} to be one-dimensional")
    if tensor.dtype not in _INDEX_DTYPES:
        raise ValueError(f"Expected {name} to have dtype int32 or int64")
