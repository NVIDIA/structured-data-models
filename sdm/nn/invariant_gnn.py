"""Permutation-invariant graph neural network modules."""

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import GELU, LayerNorm, Linear
from torch.utils.checkpoint import checkpoint
from torch_geometric.index import index2ptr
from torch_geometric.utils import coalesce, segment

_DST_CHUNK_SIZE = 8_192
_VARIANCE_FLOOR = 1e-5
_STD_DEAD_ZONE = math.sqrt(_VARIANCE_FLOOR)


class InvariantGNN(torch.nn.Module):
    r"""Apply schema-agnostic message passing to heterogeneous graphs.

    Each hop combines sum, mean, standard-deviation, minimum, and maximum
    neighborhood reductions. Relation embeddings are random unit vectors, and
    the same projections are shared across every table and relation.

    Args:
        channels: The number of input and output channels.
        device: The device.
        dtype: The dtype.
    """

    def __init__(
        self,
        channels: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs: dict[str, Any] = {
            "device": device,
            "dtype": dtype,
        }

        self.src_lin = Linear(channels, channels, **factory_kwargs)
        self.edge_type_lin = Linear(channels, channels, **factory_kwargs)

        self.skip_lin = Linear(channels, channels, **factory_kwargs)
        self.sum_lin = Linear(channels, channels, bias=False, **factory_kwargs)
        self.avg_lin = Linear(channels, channels, bias=False, **factory_kwargs)
        self.std_lin = Linear(channels, channels, bias=False, **factory_kwargs)
        self.min_lin = Linear(channels, channels, bias=False, **factory_kwargs)
        self.max_lin = Linear(channels, channels, bias=False, **factory_kwargs)

        self.norm = LayerNorm(channels, **factory_kwargs)
        self.act = GELU()

        self.post_lin = Linear(channels, channels, **factory_kwargs)
        self.post_norm = LayerNorm(channels, **factory_kwargs)

    def forward(
        self,
        x_dict: Mapping[str, Tensor],
        edge_index_dict: Mapping[tuple[str, str, str], Tensor],
        num_sampled_edges_dict: Mapping[
            tuple[str, str, str],
            Sequence[int],
        ],
        readout_table: str,
        num_hops: int,
        generator: torch.Generator,
    ) -> Tensor:
        r"""Run recurrent message passing and return one table's rows.

        Args:
            x_dict: Node features keyed by table name.
            edge_index_dict: Table-local edges keyed by relation type.
            num_sampled_edges_dict: Per-hop edge counts for each relation.
            readout_table: Table whose final node features are returned.
            num_hops: Number of recurrent message-passing hops.
            generator: Random generator for relation embeddings.

        Returns:
            Final features for ``readout_table``.
        """
        if num_hops == 0:
            return x_dict[readout_table]

        start = 0
        offset_dict: dict[str, tuple[int, int]] = {}
        for table_name, table_x in x_dict.items():
            end = start + table_x.size(0)
            offset_dict[table_name] = (start, end)
            start = end

        xs = list(x_dict.values())
        x = torch.cat(xs, dim=0) if len(xs) > 1 else xs[0]
        del xs

        edge_index_dict = {
            edge_type: edge_index[
                :,
                : sum(num_sampled_edges_dict[edge_type][:num_hops]),
            ]
            for edge_type, edge_index in edge_index_dict.items()
        }
        edge_index_dict = {
            edge_type: edge_index
            for edge_type, edge_index in edge_index_dict.items()
            if edge_index.numel() > 0
        }
        edge_index_dict = self.to_bidirectional(edge_index_dict)

        rows: list[Tensor] = []
        cols: list[Tensor] = []
        edge_types: list[Tensor] = []
        for relation_id, (edge_type, edge_index) in enumerate(
            edge_index_dict.items()
        ):
            src, _, dst = edge_type
            rows.append(edge_index[0] + offset_dict[src][0])
            cols.append(edge_index[1] + offset_dict[dst][0])
            relation_ids = edge_index.new_full(
                (edge_index.size(1),),
                relation_id,
            )
            edge_types.append(relation_ids)
            del relation_ids

        if len(rows) == 0:
            row = torch.empty(0, dtype=torch.long, device=x.device)
            col = torch.empty(0, dtype=torch.long, device=x.device)
            edge_type = torch.empty(0, dtype=torch.long, device=x.device)
        elif len(rows) == 1:
            row = rows[0]
            col = cols[0]
            edge_type = edge_types[0]
        else:
            row = torch.cat(rows, dim=0)
            col = torch.cat(cols, dim=0)
            edge_type = torch.cat(edge_types, dim=0)
        del rows
        del cols
        del edge_types

        col, perm = col.sort()
        colptr = index2ptr(col, size=x.size(0))
        row = row[perm]
        edge_type = edge_type[perm]

        edge_type_emb = torch.randn(
            (len(edge_index_dict), x.size(-1)),
            dtype=x.dtype,
            device=x.device,
            generator=generator,
        )
        edge_type_emb = F.normalize(edge_type_emb, dim=-1)
        edge_type_emb = self.edge_type_lin(edge_type_emb)

        for hop in range(num_hops):
            skip_x = self.skip_lin(x)
            x = self.src_lin(x)

            if x.size(0) <= _DST_CHUNK_SIZE:
                x = self._hop(
                    src_x=x,
                    edge_type_emb=edge_type_emb,
                    row=row,
                    edge_type=edge_type,
                    colptr=colptr,
                    skip_x=skip_x,
                )
            else:
                colptr_cpu = colptr.cpu()
                outs: list[Tensor] = []
                out: Tensor | None = None
                if not x.requires_grad:
                    out = torch.empty_like(x)

                for start in range(0, x.size(0), _DST_CHUNK_SIZE):
                    end = min(start + _DST_CHUNK_SIZE, x.size(0))
                    row_start = int(colptr_cpu[start])
                    row_end = int(colptr_cpu[end])
                    args = (
                        x,
                        edge_type_emb,
                        row[row_start:row_end],
                        edge_type[row_start:row_end],
                        colptr[start : end + 1] - colptr[start],
                        skip_x[start:end],
                    )
                    if x.requires_grad:
                        h = checkpoint(
                            self._hop,
                            *args,
                            use_reentrant=False,
                        )
                        outs.append(h)
                        del h
                    else:
                        assert out is not None
                        out[start:end] = self._hop(*args)

                if len(outs) > 0:
                    x = torch.cat(outs, dim=0)
                else:
                    assert out is not None
                    x = out

            if hop == num_hops - 1:
                readout_start, readout_end = offset_dict[readout_table]
                x = x[readout_start:readout_end]

            x = self.norm(x)
            x = self.act(x)

        x = self.post_lin(x)
        x = self.post_norm(x)

        return x  # noqa: RET504

    def _hop(
        self,
        src_x: Tensor,
        edge_type_emb: Tensor,
        row: Tensor,
        edge_type: Tensor,
        colptr: Tensor,
        skip_x: Tensor,
    ) -> Tensor:
        x_src = src_x[row] + edge_type_emb[edge_type]
        h = segment(x_src, colptr, reduce="sum")
        out = skip_x + self.sum_lin(h)
        h = h / colptr.diff().clamp(min=1).view(-1, 1)
        out = out + self.avg_lin(h)
        h = segment(x_src.square(), colptr, reduce="mean") - h.square()
        h = h.clamp(min=_VARIANCE_FLOOR).sqrt()
        h = h.masked_fill(h <= _STD_DEAD_ZONE, 0.0)
        out = out + self.std_lin(h)
        h = segment(x_src, colptr, reduce="min")
        out = out + self.min_lin(h)
        h = segment(x_src, colptr, reduce="max")
        out = out + self.max_lin(h)
        return out  # noqa: RET504

    @staticmethod
    def to_bidirectional(
        edge_index_dict: Mapping[tuple[str, str, str], Tensor],
    ) -> dict[tuple[str, str, str], Tensor]:
        r"""Add reverse relations and merge explicitly paired directions."""
        out_dict: dict[tuple[str, str, str], Tensor] = {}
        for edge_type, edge_index in edge_index_dict.items():
            if edge_type in out_dict:
                continue

            src, relation, dst = edge_type
            if relation.startswith("rev_"):
                reverse_edge_type = (dst, relation[4:], src)
            else:
                reverse_edge_type = (dst, f"rev_{relation}", src)

            if reverse_edge_type not in edge_index_dict:
                out_dict[edge_type] = edge_index
                out_dict[reverse_edge_type] = edge_index.flip(0)
            else:
                reverse_edge_index = edge_index_dict[reverse_edge_type]
                edge_index = torch.cat(
                    (edge_index, reverse_edge_index.flip(0)),
                    dim=1,
                )
                edge_index = coalesce(
                    edge_index,
                    reduce="any",
                    sort_by_row=False,
                )
                out_dict[edge_type] = edge_index
                out_dict[reverse_edge_type] = edge_index.flip(0)

        return out_dict
