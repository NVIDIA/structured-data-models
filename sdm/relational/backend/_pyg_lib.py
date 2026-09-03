# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Mapping, Sequence
from typing import Literal

import torch
from torch import Tensor

from sdm import TableTensor
from sdm.relational import RelationalData, TaskLink
from sdm.relational.join import join_index


class PyGLibRelationalSampler:
    def __init__(
        self,
        data: RelationalData,
        time_columns: Mapping[str, str],
    ) -> None:

        if not data.is_cpu:
            raise ValueError(
                f"{self.__class__.__name__!r} requires input data on CPU "
                f"(got '{data.device}')"
            )

        try:
            import pyg_lib  # noqa
        except ImportError as e:
            torch_version = torch.__version__.split("+", maxsplit=1)[0]
            if not all(part.isdigit() for part in torch_version.split(".")):
                raise ImportError(
                    "No module named 'pyg_lib'. Pre-built pyg-lib wheels are "
                    "only published for stable PyTorch releases. Please "
                    "install a stable PyTorch release or build pyg-lib "
                    "from source (see 'https://github.com/pyg-team/pyg-lib' "
                    "for more information)"
                ) from e
            if torch.version.cuda is None:
                cuda_version = "cpu"
            else:
                cuda_version = f"cu{torch.version.cuda.replace('.', '')}"
            raise ImportError(
                f"No module named 'pyg_lib'. Please install it via "
                f"'pip install pyg-lib -f https://data.pyg.org/whl/"
                f"torch-{torch_version}+{cuda_version}.html' (see "
                "'https://github.com/pyg-team/pyg-lib' for more information)"
            ) from e

        self.data = data
        self.time_columns = time_columns

        self._row_dict: dict[tuple[str, str, str], Tensor] = {}
        self._colptr_dict: dict[tuple[str, str, str], Tensor] = {}
        self._time_dict: dict[str, Tensor] = {}
        for table_name, time_column in self.time_columns.items():
            time = self.data.tables[table_name][time_column].datetime
            self._time_dict[table_name] = time.squeeze(-1).contiguous()

        for i, (rel, edge_index) in enumerate(
            zip(self.data.relationships, self.data.edge_indices())
        ):
            edge_type = (rel.left_table, str(2 * i), rel.right_table)
            self._row_dict[edge_type], self._colptr_dict[edge_type] = _to_csc(
                edge_index=edge_index,
                num_dst_nodes=self.data.tables[rel.right_table].size(0),
                src_time=self._time_dict.get(rel.left_table),
            )
            edge_type = (rel.right_table, str(2 * i + 1), rel.left_table)
            self._row_dict[edge_type], self._colptr_dict[edge_type] = _to_csc(
                edge_index=edge_index.flip(0),
                num_dst_nodes=self.data.tables[rel.left_table].size(0),
                src_time=self._time_dict.get(rel.right_table),
            )

    def sample(
        self,
        task_table: TableTensor,
        task_link: TaskLink,
        num_neighbors: Sequence[int],
        task_time_column: str | None = None,
        temporal_strategy: Literal["last", "uniform"] = "last",
    ) -> dict[str, tuple[Tensor, Tensor]]:

        if not task_table.is_cpu:
            raise NotImplementedError(
                f"{self.__class__.__name__!r} requires input data on CPU "
                f"(got '{task_table.device}')"
            )

        # Resolve entity table node indices:
        task_index, seed = join_index(
            left_table=task_table,
            right_table=self.data.tables[task_link.table],
            left_keys=task_link.task_columns,
            right_keys=task_link.table_columns,
            device=task_table.device,
        )
        task_index, perm = task_index.sort()
        seed = seed[perm]

        expected = torch.arange(
            task_table.size(-2),
            dtype=task_index.dtype,
            device=task_index.device,
        )
        if not task_index.equal(expected):
            raise ValueError(
                f"Expected each task row to match exactly one row in "
                f"{task_link.table!r}"
            )

        if task_time_column is not None:
            seed_time = task_table[task_time_column].datetime.squeeze(-1)
        else:
            fill_value = torch.iinfo(torch.int64).max
            seed_time = torch.full_like(seed, fill_value)

        # Perform subgraph sampling:
        _, _, node_dict, *_ = torch.ops.pyg.hetero_neighbor_sample(
            node_types=list(self.data.tables),
            edge_types=list(self._colptr_dict),
            rowptr_dict={
                "__".join(edge_type): colptr
                for edge_type, colptr in self._colptr_dict.items()
            },
            col_dict={
                "__".join(edge_type): row
                for edge_type, row in self._row_dict.items()
            },
            seed_dict={task_link.table: seed},
            num_neighbors_dict={
                "__".join(edge_type): list(num_neighbors)
                for edge_type in self._row_dict
            },
            node_time_dict=self._time_dict,
            edge_time_dict=None,
            seed_time_dict={task_link.table: seed_time},
            edge_weight_dict=None,
            csc=True,
            replace=False,
            directed=True,
            disjoint=True,
            temporal_strategy=temporal_strategy,
            return_edge_id=False,
        )

        return {
            table_name: tuple(node.t().contiguous())
            for table_name, node in node_dict.items()
            if node.numel() > 0
        }


def _to_csc(
    edge_index: Tensor,
    num_dst_nodes: int,
    src_time: Tensor | None = None,
) -> tuple[Tensor, Tensor]:

    if src_time is None:  # Sort primarily by destination node:
        perm = edge_index[1].argsort()
    else:  # Sort secondarily by source timestamp:
        perm = src_time[edge_index[0]].argsort()
        edge_index = edge_index[:, perm]
        perm = edge_index[1].argsort(stable=True)
    edge_index = edge_index[:, perm]

    row, col = edge_index
    colptr = torch._convert_indices_from_coo_to_csr(
        col, num_dst_nodes, out_int32=col.dtype != torch.int64
    )

    return row, colptr
