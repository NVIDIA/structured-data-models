"""Table-hop row encoding for KumoRFM."""

from collections.abc import Mapping, Sequence

import torch
from torch import Tensor

from sdm.models.tabiclv2.row_embedding import RowEmbedding
from sdm.tensor.related_tables import HomogeneousGraph


class TableHopEncoder(torch.nn.Module):
    r"""Encode related-table hops in homogeneous graph row order.

    For each table, concatenating its hop tensors along rows must reproduce the
    graph-local rows in the table block beginning at
    ``graph.node_offsets[table]``. Mapping order is otherwise irrelevant. Each
    nonempty retained hop is encoded independently without caching.

    Args:
        row_embedding: The task-specialized generic row embedding.
    """

    def __init__(self, row_embedding: RowEmbedding) -> None:
        super().__init__()
        self.row_embedding = row_embedding

    def forward(
        self,
        table_hops: Mapping[str, Sequence[Tensor]],
        y: Tensor,
        *,
        graph: HomogeneousGraph,
        node_y: Tensor | None = None,
        max_train: int | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, Tensor]:
        r"""Return global node encodings and the retained-table node mask."""
        feature = _validate_inputs(table_hops, y, graph, node_y)
        node_batch = graph.node_batch
        offsets = tuple(graph.node_offsets.items())
        context = (node_batch >= 0) & (node_batch < y.numel())
        channels = self.row_embedding.lin.out_features
        channels *= self.row_embedding.readout_token.size(-2)

        blocks: list[Tensor | None] = []
        block_rows: list[int] = []
        active_parts: list[Tensor] = []
        output_prototype: Tensor | None = None
        for i, (table, start) in enumerate(offsets):
            end = offsets[i + 1][1] if i + 1 < len(offsets) else graph.num_rows
            table_context = context[start:end]
            active = bool(table_context.any())
            block_rows.append(end - start)
            active_parts.append(
                node_batch.new_full((end - start,), active, dtype=torch.bool)
            )
            if not active:
                blocks.append(None)
                continue

            encoded_parts: list[Tensor] = []
            row = start
            for part in table_hops[table]:
                next_row = row + part.size(0)
                if part.size(0) > 0:
                    train_mask = context[row:next_row]
                    if node_y is None:
                        targets = y[node_batch[row:next_row][train_mask]]
                    else:
                        targets = node_y[row:next_row][train_mask]
                    encoded_parts.append(
                        self.row_embedding(
                            part,
                            targets,
                            train_mask=train_mask,
                            max_train=max_train,
                            generator=generator,
                            fallback_to_all=not bool(train_mask.any()),
                        )
                    )
                row = next_row

            block = torch.cat(encoded_parts, dim=0)
            blocks.append(block)
            if output_prototype is None:
                output_prototype = block

        if output_prototype is None:
            prototype = self.row_embedding.lin.weight
            if feature is not None:
                prototype = feature
            output_prototype = prototype.new_empty((0, channels))

        node_x = (
            torch.cat(
                [
                    block
                    if block is not None
                    else output_prototype.new_zeros((rows, channels))
                    for block, rows in zip(blocks, block_rows)
                ],
                dim=0,
            )
            if blocks
            else output_prototype
        )
        active_node = (
            torch.cat(active_parts)
            if active_parts
            else node_batch.new_empty(0, dtype=torch.bool)
        )
        return node_x, active_node


def _validate_inputs(
    table_hops: Mapping[str, Sequence[Tensor]],
    y: Tensor,
    graph: HomogeneousGraph,
    node_y: Tensor | None,
) -> Tensor | None:
    if not isinstance(table_hops, Mapping):
        raise TypeError("`table_hops` must be a mapping")
    if set(table_hops) != set(graph.node_offsets):
        raise ValueError("Table-hop keys must exactly match graph tables")
    if not isinstance(y, Tensor) or y.dim() != 1:
        raise ValueError("`y` must be a one-dimensional tensor")
    node_batch = graph.node_batch
    if y.device != node_batch.device:
        raise ValueError("`y` and `node_batch` must share a device")
    if node_y is not None and (
        not isinstance(node_y, Tensor)
        or node_y.dim() != 1
        or node_y.numel() != graph.num_rows
        or node_y.device != node_batch.device
    ):
        raise ValueError("`node_y` must be a vector with one value per node")

    offsets = tuple(graph.node_offsets.items())
    feature: Tensor | None = None
    for i, (table, start) in enumerate(offsets):
        end = offsets[i + 1][1] if i + 1 < len(offsets) else graph.num_rows
        parts = table_hops[table]
        if not isinstance(parts, Sequence) or len(parts) == 0:
            raise ValueError(f"Table {table!r} needs a nonempty hop sequence")
        rows = 0
        for part in parts:
            if (
                not isinstance(part, Tensor)
                or part.dim() != 2
                or not part.is_floating_point()
                or part.size(1) == 0
            ):
                raise ValueError(
                    "Hops must be 2D floating tensors with positive width"
                )
            if part.device != node_batch.device or (
                feature is not None and part.dtype != feature.dtype
            ):
                raise ValueError("Hops must share a device and dtype")
            feature = part
            rows += part.size(0)
        if rows != end - start:
            raise ValueError(
                f"Table {table!r} rows do not match its graph block"
            )
    return feature
