"""Processed-tensor KumoRFM entity prediction core."""

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor
from torch.nn import GELU, Linear, Sequential

from sdm.models.kumorfm.table_hop_encoder import TableHopEncoder
from sdm.models.tabiclv2.icl import ICLBlock
from sdm.models.tabiclv2.row_embedding import RowEmbedding
from sdm.nn import InvariantGNN
from sdm.tensor.related_tables import HomogeneousGraph


class KumoRFM(torch.nn.Module):
    r"""Predict entity targets from processed related-table tensors.

    Args:
        num_classes: Classification output width, or zero for regression.
        num_quantiles: Regression output width, or zero for classification.
        cell_channels: Width of each row readout token.
        channels: Row, graph, and dataset embedding width.
        num_row_layers: Number of row-embedding transformer layers.
        num_row_heads: Number of row-embedding attention heads.
        group_size: Number of cyclically grouped feature values per token.
        num_inducing_points: Number of induced column-attention tokens.
        num_icl_layers: Number of dataset-level ICL transformer layers.
        num_icl_heads: Number of dataset-level ICL attention heads.
        norm_bias: Whether transformer normalization layers use bias.
        dst_chunk_size: Maximum GNN destination rows aggregated at once.
        device: Parameter device.
        dtype: Parameter dtype.
    """

    def __init__(
        self,
        num_classes: int = 10,
        num_quantiles: int = 0,
        cell_channels: int = 128,
        channels: int = 512,
        num_row_layers: int = 3,
        num_row_heads: int = 8,
        group_size: int = 3,
        num_inducing_points: int = 128,
        num_icl_layers: int = 12,
        num_icl_heads: int = 8,
        norm_bias: bool = True,
        dst_chunk_size: int | None = 8192,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        _validate_non_negative_int("num_classes", num_classes)
        _validate_non_negative_int("num_quantiles", num_quantiles)
        if (num_classes > 0) == (num_quantiles > 0):
            raise ValueError(
                "Exactly one of `num_classes` and `num_quantiles` must be "
                "positive"
            )
        for name, value in (
            ("cell_channels", cell_channels),
            ("channels", channels),
            ("num_row_layers", num_row_layers),
            ("num_row_heads", num_row_heads),
            ("group_size", group_size),
            ("num_inducing_points", num_inducing_points),
            ("num_icl_layers", num_icl_layers),
            ("num_icl_heads", num_icl_heads),
        ):
            _validate_positive_int(name, value)
        if channels % cell_channels != 0:
            raise ValueError("`channels` must be divisible by `cell_channels`")

        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}
        row_embedding = RowEmbedding(
            num_classes=num_classes,
            channels=cell_channels,
            num_layers=num_row_layers,
            num_heads=num_row_heads,
            group_size=group_size,
            num_inducing_points=num_inducing_points,
            num_readout_tokens=channels // cell_channels,
            norm_bias=norm_bias,
            **factory_kwargs,
        )
        self.table_hop_encoder = TableHopEncoder(row_embedding)
        self.gnn = InvariantGNN(
            channels=channels,
            dst_chunk_size=dst_chunk_size,
            **factory_kwargs,
        )
        self.icl_block = ICLBlock(
            num_classes=num_classes,
            channels=channels,
            num_layers=num_icl_layers,
            num_heads=num_icl_heads,
            norm_bias=norm_bias,
            **factory_kwargs,
        )
        output_channels = num_classes or num_quantiles
        self.head = Sequential(
            Linear(channels, 2 * channels, **factory_kwargs),
            GELU(),
            Linear(2 * channels, output_channels, **factory_kwargs),
        )
        self.num_classes = num_classes
        self.num_quantiles = num_quantiles

    def forward(
        self,
        table_hops: Mapping[str, Sequence[Tensor]],
        y: Tensor,
        *,
        graph: HomogeneousGraph,
        entity_relationship: int,
        max_train: int | None = 20_000,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        r"""Predict non-context entities in ascending task-ID order."""
        root_index = _validate_forward_inputs(
            y=y,
            graph=graph,
            entity_relationship=entity_relationship,
            num_classes=self.num_classes,
        )
        node_x, active_node = self.table_hop_encoder(
            table_hops,
            y,
            graph=graph,
            max_train=max_train,
            generator=generator,
        )
        active_edge = active_node[graph.edge_index].all(dim=0)
        node_x = self.gnn(
            x=node_x,
            edge_index=graph.edge_index[:, active_edge],
            edge_type=graph.edge_type[active_edge],
            num_edge_types=graph.num_edge_types,
            num_hops=max(len(parts) for parts in table_hops.values()) - 1,
            generator=generator,
        )
        node_x = node_x.masked_fill(~active_node.unsqueeze(-1), 0.0)
        if not bool(active_node[root_index].all()):
            raise ValueError("All selected entity roots must be active")
        entity_x = node_x.index_select(0, root_index)
        return self.head(self.icl_block(entity_x, y))


def _validate_forward_inputs(
    *,
    y: Tensor,
    graph: HomogeneousGraph,
    entity_relationship: int,
    num_classes: int,
) -> Tensor:
    _validate_non_negative_int("entity_relationship", entity_relationship)
    _validate_non_negative_int("graph.num_task_rows", graph.num_task_rows)
    _validate_non_negative_int("graph.num_rows", graph.num_rows)
    if entity_relationship not in graph.task_edge_indices:
        raise ValueError("`entity_relationship` is absent from the graph")

    node_batch = graph.node_batch
    if (
        not isinstance(node_batch, Tensor)
        or node_batch.dtype != torch.long
        or node_batch.dim() != 1
        or node_batch.numel() != graph.num_rows
    ):
        raise ValueError("`graph.node_batch` must be a long vector per node")
    task_edges = graph.task_edge_indices[entity_relationship]
    if (
        not isinstance(task_edges, Tensor)
        or task_edges.dtype != torch.long
        or task_edges.dim() != 2
        or task_edges.size(0) != 2
        or task_edges.device != node_batch.device
    ):
        raise ValueError("Task edges must have shape [2, E] on graph device")

    num_tasks = graph.num_task_rows
    task_id, root_index = task_edges
    order = task_id.argsort(stable=True)
    expected = torch.arange(num_tasks, device=task_id.device)
    if task_id.numel() != num_tasks or not torch.equal(
        task_id[order], expected
    ):
        raise ValueError("Entity task edges must contain each task ID once")
    root_index = root_index[order]
    if root_index.unique().numel() != num_tasks:
        raise ValueError("Entity roots must be unique")
    if bool(((root_index < 0) | (root_index >= graph.num_rows)).any()):
        raise ValueError("Entity roots contain an out-of-range node ID")
    if not torch.equal(node_batch[root_index], expected):
        raise ValueError("Entity root node_batch IDs must match task IDs")
    assigned = node_batch >= 0
    if bool((node_batch[assigned] >= num_tasks).any()):
        raise ValueError("Assigned node_batch IDs must be below num_task_rows")

    if not isinstance(y, Tensor) or y.dim() != 1 or y.numel() == 0:
        raise ValueError("`y` must be a nonempty one-dimensional tensor")
    if y.device != node_batch.device:
        raise ValueError("`y` must be on the graph device")
    if y.numel() > num_tasks:
        raise ValueError("`y` cannot contain more values than task rows")
    if num_classes == 0:
        if not y.is_floating_point():
            raise TypeError("Regression targets must have a floating dtype")
    elif y.is_floating_point() or y.is_complex():
        raise TypeError("Classification targets must have an integral dtype")
    elif bool(((y < 0) | (y >= num_classes)).any()):
        raise ValueError(
            f"Classification targets must be in [0, {num_classes})"
        )
    return root_index


def _validate_positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"`{name}` must be an integer")
    if value < 1:
        raise ValueError(f"`{name}` must be positive")


def _validate_non_negative_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"`{name}` must be an integer")
    if value < 0:
        raise ValueError(f"`{name}` must be non-negative")
