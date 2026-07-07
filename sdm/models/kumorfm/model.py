"""Processed-tensor KumoRFM prediction core."""

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor
from torch.nn import GELU, Linear, Sequential
from torch.utils.checkpoint import checkpoint

from sdm.models.kumorfm.table_hop_encoder import TableHopEncoder
from sdm.models.tabiclv2.icl import ICLBlock
from sdm.models.tabiclv2.row_embedding import RowEmbedding
from sdm.nn import InvariantGNN
from sdm.tensor.related_tables import HomogeneousGraph

_LINK_PREDICTION_CHUNK_SIZE = 1000


class KumoRFM(torch.nn.Module):
    r"""Predict entity or link targets from processed related-table tensors.

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
        node_x, active_node = self._encode_graph(
            table_hops,
            y,
            graph=graph,
            max_train=max_train,
            generator=generator,
        )
        if not bool(active_node[root_index].all()):
            raise ValueError("All selected entity roots must be active")
        entity_x = node_x.index_select(0, root_index)
        return self._predict(entity_x, y)

    def forward_link_prediction(
        self,
        table_hops: Mapping[str, Sequence[Tensor]],
        y: Tensor,
        y_lp: Tensor,
        *,
        graph: HomogeneousGraph,
        readout_table: str,
        context_node_index: Tensor,
        candidate_node_index: Tensor,
        max_lp_context_size: int | None = None,
        max_train: int | None = 20_000,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        r"""Predict binary links for explicit global candidate node IDs."""
        context_node_index, y_lp = _validate_link_prediction_inputs(
            y,
            y_lp,
            graph,
            readout_table,
            context_node_index,
            candidate_node_index,
            self.num_classes,
            max_lp_context_size,
            max_train,
            generator,
        )

        node_batch = graph.node_batch
        contextual = (node_batch >= 0) & (node_batch < y.numel())
        node_y = y.new_zeros(graph.num_rows)
        node_y[contextual] = y[node_batch[contextual]]
        node_y[context_node_index] = y_lp.to(dtype=y.dtype)

        node_x, _ = self._encode_graph(
            table_hops,
            y,
            graph=graph,
            node_y=node_y,
            max_train=max_train,
            generator=generator,
        )
        context_x = node_x.index_select(0, context_node_index)
        candidate_x = node_x.index_select(0, candidate_node_index)
        if (
            max_lp_context_size is not None
            and context_x.size(0) > max_lp_context_size
        ):
            index = torch.randperm(
                context_x.size(0),
                device=context_x.device,
                generator=generator,
            )[:max_lp_context_size]
            context_x = context_x.index_select(0, index)
            y_lp = y_lp.index_select(0, index)

        if candidate_x.size(0) == 0:
            prediction_x = torch.cat((context_x, candidate_x))
            return self._predict(prediction_x, y_lp)[..., :2]

        outputs: list[Tensor] = []
        for candidate_chunk in candidate_x.split(_LINK_PREDICTION_CHUNK_SIZE):
            prediction_x = torch.cat((context_x, candidate_chunk))
            if torch.is_grad_enabled():
                logits = checkpoint(
                    self._predict,
                    prediction_x,
                    y_lp,
                    use_reentrant=False,
                )
            else:
                logits = self._predict(prediction_x, y_lp)
            outputs.append(logits[..., :2])
        return torch.cat(outputs, dim=0) if len(outputs) > 1 else outputs[0]

    def _encode_graph(
        self,
        table_hops: Mapping[str, Sequence[Tensor]],
        y: Tensor,
        *,
        graph: HomogeneousGraph,
        node_y: Tensor | None = None,
        max_train: int | None,
        generator: torch.Generator | None,
    ) -> tuple[Tensor, Tensor]:
        node_x, active_node = self.table_hop_encoder(
            table_hops,
            y,
            graph=graph,
            node_y=node_y,
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
        return node_x, active_node

    def _predict(self, x: Tensor, y: Tensor) -> Tensor:
        return self.head(self.icl_block(x, y))


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


def _validate_link_prediction_inputs(
    y: Tensor,
    y_lp: Tensor,
    graph: HomogeneousGraph,
    readout_table: str,
    context_node_index: Tensor,
    candidate_node_index: Tensor,
    num_classes: int,
    max_lp_context_size: int | None,
    max_train: int | None,
    generator: torch.Generator | None,
) -> tuple[Tensor, Tensor]:
    if num_classes < 2:
        raise ValueError(
            "Link prediction requires at least two native classes"
        )
    device = graph.node_batch.device
    for name, value in (("y", y), ("y_lp", y_lp)):
        if not isinstance(value, Tensor):
            raise TypeError(f"`{name}` must be a tensor")
        if value.dim() != 1:
            raise ValueError(f"`{name}` must be one-dimensional")
        if value.device != device:
            raise ValueError(f"`{name}` must be on the graph device")
        if value.is_floating_point() or value.is_complex():
            raise TypeError(f"`{name}` must have an integral or bool dtype")
    if y.numel() == 0:
        raise ValueError("`y` must be nonempty")
    if y.numel() > graph.num_task_rows:
        raise ValueError("`y` cannot contain more values than task rows")
    if bool(((y < 0) | (y >= num_classes)).any()):
        raise ValueError(f"`y` values must be in [0, {num_classes})")
    if bool(((y_lp != 0) & (y_lp != 1)).any()):
        raise ValueError("`y_lp` must contain only binary labels 0 and 1")
    if max_train is not None:
        _validate_positive_int("max_train", max_train)
    if max_lp_context_size is not None:
        _validate_positive_int("max_lp_context_size", max_lp_context_size)
    if generator is not None:
        if not isinstance(generator, torch.Generator):
            raise TypeError("`generator` must be a torch.Generator or None")
        if generator.device != device:
            raise ValueError("`generator` must be on the graph device")
    if not isinstance(readout_table, str):
        raise TypeError("`readout_table` must be a string")
    if readout_table not in graph.node_offsets:
        raise ValueError("`readout_table` is absent from graph node blocks")
    offsets = tuple(graph.node_offsets.items())
    block = tuple(graph.node_offsets).index(readout_table)
    start = graph.node_offsets[readout_table]
    end = offsets[block + 1][1] if block + 1 < len(offsets) else graph.num_rows

    for name, value in (
        ("context_node_index", context_node_index),
        ("candidate_node_index", candidate_node_index),
    ):
        if not isinstance(value, Tensor):
            raise TypeError(f"`{name}` must be a tensor")
        if value.dtype != torch.long or value.dim() != 1:
            raise ValueError(f"`{name}` must be a one-dimensional long tensor")
        if value.device != device:
            raise ValueError(f"`{name}` must be on the graph device")
        if bool(((value < 0) | (value >= graph.num_rows)).any()):
            raise ValueError(f"`{name}` contains an out-of-range node ID")
        if value.unique().numel() != value.numel():
            raise ValueError(f"`{name}` must contain unique node IDs")
    if y_lp.numel() != context_node_index.numel():
        raise ValueError("`y_lp` must align with `context_node_index`")

    order = context_node_index.argsort(stable=True)
    context_node_index = context_node_index.index_select(0, order)
    y_lp = y_lp.index_select(0, order)
    readout_nodes = torch.arange(start, end, device=device)
    readout_batch = graph.node_batch[start:end]
    expected_context = readout_nodes[
        (readout_batch >= 0) & (readout_batch < y.numel())
    ]
    if expected_context.numel() == 0:
        raise ValueError("Readout context must be nonempty")
    if not torch.equal(context_node_index, expected_context):
        raise ValueError("Context IDs must exactly match the readout context")

    if bool(
        ((candidate_node_index < start) | (candidate_node_index >= end)).any()
    ):
        raise ValueError("Candidate node IDs must be in the readout block")
    combined = torch.cat((context_node_index, candidate_node_index))
    if combined.unique().numel() != combined.numel():
        raise ValueError("Context and candidate node IDs must be disjoint")
    candidate_batch = graph.node_batch[candidate_node_index]
    invalid = (candidate_batch < y.numel()) | (
        candidate_batch >= graph.num_task_rows
    )
    if bool(invalid.any()):
        raise ValueError("Candidate node_batch IDs must identify task rows")
    return context_node_index, y_lp


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
