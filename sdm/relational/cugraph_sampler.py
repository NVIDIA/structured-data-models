from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
from torch import Tensor

from sdm import NaT, Stype, TableTensor
from sdm.relational.data import RelationalData
from sdm.relational.join import join_index
from sdm.relational.sampler import (
    RelationalSampler,
    RelationalSamplerOutput,
    TemporalSamplingConfig,
    _validate_time_columns,
)
from sdm.relational.task import TaskLink

_INTEGER_DTYPES = {
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
}


class CuGraphRelationalSampler(RelationalSampler):
    r"""GPU subgraph sampler over relational data.

    The sampler materializes a persistent single-GPU cuGraph topology and
    keeps task lookup, sampling, and output assembly on the CUDA device.
    For temporal data, each hop samples uniformly from eligible neighbors
    against the original task cutoff. This retains a fixed cutoff across hops
    instead of propagating sampled edge times.

    Args:
        data: CUDA-resident tables and their relationships.
        temporal: Temporal sampling configuration. Only uniform neighbor
            selection is currently supported.
        random_state: Seed for the advancing cuGraph random-state stream.
    """

    def __init__(
        self,
        data: RelationalData,
        temporal: TemporalSamplingConfig | None = None,
        random_state: int | None = None,
    ) -> None:
        if data.device.type != "cuda":
            raise ValueError(
                f"'{self.__class__.__name__}' requires CUDA-resident data"
            )
        if temporal is not None and temporal.strategy == "last":
            raise NotImplementedError(
                "cuGraph temporal sampling does not support strategy 'last'"
            )

        self.data = data
        self.temporal = temporal
        self.time_columns = (
            temporal.time_columns if temporal is not None else {}
        )
        self._generator = np.random.default_rng(random_state)
        _validate_time_columns(self.data, self.time_columns)

        try:
            import cupy as cp
            import pylibcugraph  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "CUDA relational sampling requires cupy and pylibcugraph"
            ) from exc

        self._cp = cp
        self._pylibcugraph = pylibcugraph
        self._table_names = tuple(self.data.tables)
        self._table_ids = {
            table_name: i for i, table_name in enumerate(self._table_names)
        }
        self._numeric_seed_lookups: dict[
            tuple[str, str], tuple[Tensor, int, Tensor, Tensor]
        ] = {}

        offsets = [0]
        for table in self.data.tables.values():
            offsets.append(offsets[-1] + table.size(0))
        self._vertex_offsets = torch.tensor(
            offsets,
            dtype=torch.int64,
            device=self.data.device,
        )
        self._num_vertices = offsets[-1]
        self._num_edge_types = 2 * len(self.data.relationships)
        self._num_edges = 0
        self._outgoing_edge_types = {
            table_name: [] for table_name in self._table_names
        }
        self._neighbor_tables = {
            table_name: set() for table_name in self._table_names
        }
        for i, relationship in enumerate(self.data.relationships):
            self._outgoing_edge_types[relationship.right_table].append(2 * i)
            self._outgoing_edge_types[relationship.left_table].append(
                2 * i + 1
            )
            self._neighbor_tables[relationship.right_table].add(
                relationship.left_table
            )
            self._neighbor_tables[relationship.left_table].add(
                relationship.right_table
            )

        with torch.cuda.device(self.data.device):
            self._build_graph()

    def sample(
        self,
        task_table: TableTensor,
        task_link: TaskLink | Mapping[str, str | Sequence[str]],
        num_neighbors: Sequence[int],
        task_time_column: str | None = None,
    ) -> RelationalSamplerOutput:
        r"""Sample CUDA-resident related tables for task rows."""
        task_link = self._validate_sample_inputs(
            task_table=task_table,
            task_link=task_link,
            task_time_column=task_time_column,
        )
        if task_table.device != self.data.device:
            raise ValueError(
                "Expected task and relational tables on the same CUDA device"
            )
        if len(num_neighbors) == 0:
            raise ValueError("Expected at least one sampling hop")

        with torch.cuda.device(self.data.device):
            seed = self._resolve_seed(
                task_table=task_table,
                task_link=task_link,
            )
            if task_time_column is None:
                seed_time = torch.full_like(
                    seed,
                    torch.iinfo(torch.int64).max,
                )
            else:
                seed_time = task_table[task_time_column].datetime.squeeze(-1)

            if self._num_edges == 0:
                nodes = self._seed_nodes(seed=seed, task_link=task_link)
            elif self.time_columns:
                nodes = self._sample_temporal(
                    seed=seed,
                    seed_time=seed_time,
                    num_neighbors=num_neighbors,
                    seed_table=task_link.table,
                )
            else:
                nodes = self._sample_non_temporal(
                    seed=seed,
                    num_neighbors=num_neighbors,
                )

        return self._to_output(
            task_table=task_table,
            task_link=task_link,
            nodes=nodes,
        )

    def _build_graph(self) -> None:
        cp = self._cp
        pylibcugraph = self._pylibcugraph
        edge_indices = self.data.edge_indices(dtype=torch.int64)

        srcs: list[Tensor] = []
        dsts: list[Tensor] = []
        edge_types: list[Tensor] = []
        edge_times: list[Tensor] = []

        times = {
            table_name: self.data.tables[table_name][
                column_name
            ].datetime.squeeze(-1)
            for table_name, column_name in self.time_columns.items()
        }

        for i, (relationship, edge_index) in enumerate(
            zip(self.data.relationships, edge_indices)
        ):
            left_offset = self._table_offset(relationship.left_table)
            right_offset = self._table_offset(relationship.right_table)
            left, right = edge_index

            # cuGraph traverses outgoing edges. Reverse each PyG direction so
            # a right-table seed discovers its left-table relational rows.
            srcs.extend((right + right_offset, left + left_offset))
            dsts.extend((left + left_offset, right + right_offset))
            edge_types.extend(
                (
                    torch.full_like(left, 2 * i, dtype=torch.int32),
                    torch.full_like(left, 2 * i + 1, dtype=torch.int32),
                )
            )

            for table_name, index in (
                (relationship.left_table, left),
                (relationship.right_table, right),
            ):
                if table_name in times:
                    edge_times.append(times[table_name][index])
                else:
                    edge_times.append(torch.full_like(index, NaT))

        if self._num_edge_types == 0:
            self._resource_handle = None
            self._graph = None
            return

        src = torch.cat(srcs)
        dst = torch.cat(dsts)
        self._num_edges = src.numel()
        edge_type = torch.cat(edge_types)
        edge_time = torch.cat(edge_times) if self.time_columns else None
        self._has_outgoing = torch.zeros(
            self._num_vertices,
            dtype=torch.bool,
            device=self.data.device,
        )
        self._has_outgoing[src] = True

        self._resource_handle = pylibcugraph.ResourceHandle()
        properties = pylibcugraph.GraphProperties(
            is_multigraph=True,
            is_symmetric=False,
        )
        self._graph = pylibcugraph.SGGraph(
            self._resource_handle,
            properties,
            cp.from_dlpack(src),
            cp.from_dlpack(dst),
            vertices_array=cp.arange(self._num_vertices, dtype=cp.int64),
            edge_id_array=cp.arange(src.numel(), dtype=cp.int64),
            edge_type_array=cp.from_dlpack(edge_type),
            edge_start_time_array=(
                cp.from_dlpack(edge_time) if edge_time is not None else None
            ),
            store_transposed=False,
            renumber=False,
            do_expensive_check=False,
        )

    def _resolve_seed(
        self,
        task_table: TableTensor,
        task_link: TaskLink,
    ) -> Tensor:
        local_seed = self._resolve_numeric_seed(
            task_table=task_table,
            task_link=task_link,
        )
        if local_seed is None:
            local_seed = self._resolve_seed_join(
                task_table=task_table,
                task_link=task_link,
            )
        return local_seed + self._table_offset(task_link.table)

    def _resolve_numeric_seed(
        self,
        task_table: TableTensor,
        task_link: TaskLink,
    ) -> Tensor | None:
        if len(task_link.task_columns) != 1:
            return None

        task_value = self._id_column(
            task_table,
            task_link.task_columns[0],
        )
        table_value = self._id_column(
            self.data.tables[task_link.table],
            task_link.table_columns[0],
        )
        # ColumnarTensor stores numeric IDs as plain tensors. String and
        # composite IDs continue through the general cuDF join below.
        if (
            task_value is None
            or table_value is None
            or task_value.__class__ is not Tensor
            or table_value.__class__ is not Tensor
            or task_value.dtype != table_value.dtype
            or task_value.dtype not in _INTEGER_DTYPES
        ):
            return None

        key = (task_link.table, task_link.table_columns[0])
        lookup = self._numeric_seed_lookups.get(key)
        if lookup is None or not self._lookup_matches_source(
            lookup,
            table_value,
        ):
            value, row = table_value.sort()
            lookup = (table_value, table_value._version, value, row)
            self._numeric_seed_lookups[key] = lookup
        _, _, value, row = lookup

        task_value = task_value.contiguous()
        lower = torch.searchsorted(value, task_value)
        upper = torch.searchsorted(value, task_value, right=True)
        if not torch.all((upper - lower) == 1):
            raise ValueError(
                f"Expected each task row to match exactly one row in "
                f"'{task_link.table}'"
            )
        return row[lower]

    @staticmethod
    def _lookup_matches_source(
        lookup: tuple[Tensor, int, Tensor, Tensor],
        source: Tensor,
    ) -> bool:
        cached, version, _, _ = lookup
        return (
            cached._version == version
            and cached.dtype == source.dtype
            and cached.device == source.device
            and cached.layout == source.layout
            and cached.data_ptr() == source.data_ptr()
            and cached.size() == source.size()
            and cached.stride() == source.stride()
            and cached.storage_offset() == source.storage_offset()
        )

    def _resolve_seed_join(
        self,
        task_table: TableTensor,
        task_link: TaskLink,
    ) -> Tensor:
        task_index, seed = join_index(
            left_table=task_table,
            right_table=self.data.tables[task_link.table],
            left_keys=task_link.task_columns,
            right_keys=task_link.table_columns,
            dtype=torch.int64,
            device=task_table.device,
        )
        task_index, perm = task_index.sort()
        seed = seed[perm]

        expected = torch.arange(
            task_table.size(0),
            dtype=task_index.dtype,
            device=task_index.device,
        )
        if not task_index.equal(expected):
            raise ValueError(
                f"Expected each task row to match exactly one row in "
                f"'{task_link.table}'"
            )

        return seed

    @staticmethod
    def _id_column(table: TableTensor, column: str) -> Tensor | None:
        table = table[column]
        if table.id._validity[0] is not None:
            return None
        return table.id[..., 0]

    def _sample_non_temporal(
        self,
        seed: Tensor,
        num_neighbors: Sequence[int],
    ) -> dict[str, tuple[Tensor, Tensor]]:
        cp = self._cp
        example = torch.arange(seed.numel(), device=seed.device)
        seed_keys = example * self._num_vertices + seed
        if num_neighbors[0] == 0:
            return self._nodes_from_keys(seed_keys)

        active = self._has_outgoing[seed]
        active_seed = seed[active]
        active_example = example[active]
        if active_seed.numel() == 0:
            return self._nodes_from_keys(seed_keys)

        fanout = np.repeat(
            np.asarray(num_neighbors, dtype=np.int32),
            self._num_edge_types,
        )
        label_offsets = cp.arange(active_seed.numel() + 1, dtype=cp.int64)
        result = self._pylibcugraph.heterogeneous_uniform_neighbor_sample(
            self._resource_handle,
            self._graph,
            cp.from_dlpack(active_seed),
            label_offsets,
            cp.from_dlpack(self._vertex_offsets),
            fanout,
            num_edge_types=self._num_edge_types,
            with_replacement=False,
            do_expensive_check=False,
            prior_sources_behavior="exclude",
            deduplicate_sources=True,
            disjoint_sampling=False,
            return_hops=True,
            renumber=True,
            retain_seeds=True,
            compression="COO",
            compress_per_hop=False,
            random_state=self._next_random_state(),
        )
        node = self._as_tensor(result["renumber_map"])
        offsets = self._as_tensor(result["renumber_map_offsets"]).long()
        position = torch.arange(node.numel(), device=node.device)
        segment = torch.bucketize(position, offsets[1:], right=True)
        num_tables = len(self._table_names)
        sampled_example = active_example[
            segment.div(num_tables, rounding_mode="floor")
        ]
        sampled_keys = sampled_example * self._num_vertices + node
        return self._nodes_from_keys(
            torch.cat((seed_keys, sampled_keys)).unique(sorted=True)
        )

    def _sample_temporal(
        self,
        seed: Tensor,
        seed_time: Tensor,
        num_neighbors: Sequence[int],
        seed_table: str,
    ) -> dict[str, tuple[Tensor, Tensor]]:
        total = self._num_vertices
        example = torch.arange(seed.numel(), device=seed.device)
        visited = (example * total + seed).unique(sorted=True)
        frontier = visited
        possible_source_tables = {seed_table}

        for count in num_neighbors:
            if frontier.numel() == 0:
                break
            frontier_example = frontier.div(total, rounding_mode="floor")
            frontier_node = frontier.remainder(total)
            sampled = self._sample_temporal_hop(
                frontier_node=frontier_node,
                frontier_example=frontier_example,
                seed_time=seed_time,
                count=count,
                possible_source_tables=possible_source_tables,
            )
            if sampled.numel() == 0:
                break

            sampled = sampled.unique(sorted=True)
            position = torch.searchsorted(visited, sampled)
            clamped = position.clamp_max(max(visited.numel() - 1, 0))
            is_visited = (position < visited.numel()) & (
                visited[clamped] == sampled
            )
            frontier = sampled[~is_visited]
            visited = torch.cat((visited, frontier)).sort().values
            possible_source_tables = {
                neighbor
                for table_name in possible_source_tables
                for neighbor in self._neighbor_tables[table_name]
            }

        return self._nodes_from_keys(visited)

    def _nodes_from_keys(
        self,
        keys: Tensor,
    ) -> dict[str, tuple[Tensor, Tensor]]:
        example = keys.div(self._num_vertices, rounding_mode="floor")
        node = keys.remainder(self._num_vertices)
        table_id = torch.bucketize(
            node,
            self._vertex_offsets[1:],
            right=True,
        )
        nodes: dict[str, tuple[Tensor, Tensor]] = {}
        for i, table_name in enumerate(self._table_names):
            mask = table_id == i
            nodes[table_name] = (
                example[mask],
                node[mask] - self._vertex_offsets[i],
            )
        return nodes

    def _sample_temporal_hop(
        self,
        frontier_node: Tensor,
        frontier_example: Tensor,
        seed_time: Tensor,
        count: int,
        possible_source_tables: set[str],
    ) -> Tensor:
        cp = self._cp
        num_examples = seed_time.numel()
        counts = torch.bincount(
            frontier_example,
            minlength=num_examples,
        )
        label_offsets = torch.cat(
            (
                torch.zeros(1, dtype=torch.int64, device=self.data.device),
                counts.cumsum(0),
            )
        )

        if count == 0:
            return torch.empty(0, dtype=torch.int64, device=self.data.device)
        # cuGraph interprets fanout independently for every edge type. Avoid
        # scheduling types whose source table cannot occur in this hop.
        fanout = np.zeros(self._num_edge_types, dtype=np.int32)
        for table_name in possible_source_tables:
            fanout[self._outgoing_edge_types[table_name]] = count

        result = (
            self._pylibcugraph.heterogeneous_uniform_temporal_neighbor_sample(
                self._resource_handle,
                self._graph,
                "edge_start_time",
                cp.from_dlpack(frontier_node),
                cp.from_dlpack(seed_time[frontier_example]),
                cp.from_dlpack(label_offsets),
                cp.from_dlpack(self._vertex_offsets),
                fanout,
                num_edge_types=self._num_edge_types,
                with_replacement=False,
                do_expensive_check=False,
                prior_sources_behavior=None,
                deduplicate_sources=True,
                disjoint_sampling=False,
                return_hops=False,
                renumber=False,
                retain_seeds=False,
                compression="COO",
                compress_per_hop=False,
                random_state=self._next_random_state(),
                temporal_sampling_comparison="monotonically_decreasing",
            )
        )
        minor = self._as_tensor(result["minors"])
        if minor.numel() == 0:
            return minor
        batch = self._as_tensor(result["batch_id"]).long()

        return batch * self._num_vertices + minor

    def _seed_nodes(
        self,
        seed: Tensor,
        task_link: TaskLink,
    ) -> dict[str, tuple[Tensor, Tensor]]:
        local_seed = seed - self._table_offset(task_link.table)
        return {
            task_link.table: (
                torch.arange(seed.numel(), device=seed.device),
                local_seed,
            )
        }

    def _table_offset(self, table_name: str) -> Tensor:
        return self._vertex_offsets[self._table_ids[table_name]]

    def _next_random_state(self) -> int:
        return int(
            self._generator.integers(
                low=0,
                high=np.iinfo(np.int32).max,
            )
        )

    @staticmethod
    def _as_tensor(array: Any) -> Tensor:
        return torch.from_dlpack(array)
