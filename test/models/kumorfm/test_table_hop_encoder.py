from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import pytest
import torch
from sdm.cache import Cache
from sdm.models.kumorfm import TableHopEncoder
from sdm.models.tabiclv2.row_embedding import RowEmbedding
from sdm.tensor.related_tables import HomogeneousGraph
from sdm.testing import withCUDA
from torch import Tensor


@dataclass
class _Call:
    width: int
    y: Tensor
    train_mask: Tensor
    fallback_to_all: bool
    max_train: int | None
    generator: torch.Generator | None


class _RecordingRowEmbedding(RowEmbedding):
    def __init__(
        self,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float64,
    ) -> None:
        torch.nn.Module.__init__(self)
        self.lin = torch.nn.Linear(
            1, 2, bias=False, device=device, dtype=dtype
        )
        with torch.no_grad():
            self.lin.weight.copy_(
                torch.tensor([[1.0], [2.0]], device=device, dtype=dtype)
            )
        self.readout_token = torch.nn.Parameter(
            torch.zeros(1, 2, device=device, dtype=dtype)
        )
        self.calls: list[_Call] = []

    def forward(
        self,
        x: Tensor,
        y: Tensor,
        *,
        train_mask: Tensor | None = None,
        cache: Cache | None = None,
        max_train: int | None = None,
        generator: torch.Generator | None = None,
        fallback_to_all: bool = False,
    ) -> Tensor:
        assert train_mask is not None
        assert cache is None
        self.calls.append(
            _Call(
                x.size(1),
                y.detach().clone(),
                train_mask.clone(),
                fallback_to_all,
                max_train,
                generator,
            )
        )
        targets = x.new_zeros((x.size(0), 1)).masked_scatter(
            train_mask.unsqueeze(1), y.to(x.dtype)
        )
        return self.lin(x[:, :1]) + targets


def _graph(
    offsets: Mapping[str, int],
    node_batch: Tensor,
) -> HomogeneousGraph:
    device = node_batch.device
    return HomogeneousGraph(
        num_rows=node_batch.numel(),
        num_task_rows=int(node_batch.max()) + 1,
        node_offsets=dict(offsets),
        edge_index=torch.empty((2, 0), dtype=torch.long, device=device),
        edge_type=torch.empty(0, dtype=torch.long, device=device),
        num_edge_types=0,
        task_edge_indices={},
        node_batch=node_batch,
        num_hops=0,
    )


def test_table_hop_encoding_alignment_fallback_and_node_targets() -> None:
    dtype = torch.float64
    table_hops = {
        "inactive": [torch.tensor([[5.0], [6.0]], dtype=dtype)],
        "active": [
            torch.tensor([[1.0, 9.0], [2.0, 8.0]], dtype=dtype),
            torch.empty((0, 4), dtype=dtype),
            torch.tensor([[3.0]], dtype=dtype),
        ],
    }
    node_batch = torch.tensor([0, -1, 2, 2, -1])
    graph = _graph({"active": 0, "inactive": 3}, node_batch)
    row_embedding = _RecordingRowEmbedding(dtype=dtype)
    encoder = TableHopEncoder(row_embedding)
    generator = torch.Generator().manual_seed(7)

    node_x, active = encoder(
        table_hops,
        torch.tensor([10.0], dtype=dtype),
        graph=graph,
        max_train=3,
        generator=generator,
    )

    torch.testing.assert_close(
        node_x,
        torch.tensor(
            [[11.0, 12.0], [2.0, 4.0], [3.0, 6.0], [0.0, 0.0], [0.0, 0.0]],
            dtype=dtype,
        ),
    )
    assert active.tolist() == [True, True, True, False, False]
    assert [call.width for call in row_embedding.calls] == [2, 1]
    assert [call.y.tolist() for call in row_embedding.calls] == [[10.0], []]
    assert [call.train_mask.tolist() for call in row_embedding.calls] == [
        [True, False],
        [False],
    ]
    assert [call.fallback_to_all for call in row_embedding.calls] == [
        False,
        True,
    ]
    assert all(call.max_train == 3 for call in row_embedding.calls)
    assert all(call.generator is generator for call in row_embedding.calls)

    row_embedding.calls.clear()
    encoder(
        table_hops,
        torch.tensor([10.0], dtype=dtype),
        graph=graph,
        node_y=torch.tensor([21.0, 22.0, 23.0, 24.0, 25.0], dtype=dtype),
    )
    assert [call.y.tolist() for call in row_embedding.calls] == [[21.0], []]


@pytest.mark.parametrize(
    ("parts", "num_rows"),
    [
        ([], 2),
        ([torch.ones(2)], 2),
        ([torch.ones(2, 0)], 2),
        ([torch.ones(2, 2, dtype=torch.long)], 2),
        ([torch.ones(1, 2), torch.ones(1, 2, dtype=torch.float64)], 2),
        ([torch.ones(1, 2)], 2),
        ([torch.ones(1, 2), torch.empty(1, 2, device="meta")], 2),
    ],
)
def test_rejects_invalid_hop_boundaries(
    parts: Sequence[Tensor],
    num_rows: int,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        TableHopEncoder(_RecordingRowEmbedding(dtype=torch.float32))(
            {"table": parts},
            torch.ones(1),
            graph=_graph(
                {"table": 0},
                torch.zeros(num_rows, dtype=torch.long),
            ),
        )


@pytest.mark.parametrize(
    "case",
    ["keys", "y", "node_y_shape", "node_y_device"],
)
def test_rejects_invalid_vector_boundaries(case: str) -> None:
    table_hops = {"table": [torch.ones(2, 2, dtype=torch.float64)]}
    y = torch.ones(1, dtype=torch.float64)
    node_batch = torch.zeros(2, dtype=torch.long)
    node_y = None
    if case == "keys":
        table_hops = {}
    elif case == "y":
        y = y.unsqueeze(1)
    elif case == "node_y_shape":
        node_y = torch.ones(1, dtype=torch.float64)
    else:
        node_y = torch.ones(2, device="meta")

    with pytest.raises((TypeError, ValueError)):
        TableHopEncoder(_RecordingRowEmbedding())(
            table_hops,
            y,
            graph=_graph({"table": 0}, node_batch),
            node_y=node_y,
        )


@withCUDA
def test_preserves_gradients_device_and_dtype(device: torch.device) -> None:
    dtype = torch.float64
    x = torch.randn(2, 3, device=device, dtype=dtype, requires_grad=True)
    y = torch.randn(2, device=device, dtype=dtype, requires_grad=True)
    row_embedding = _RecordingRowEmbedding(device=device, dtype=dtype)

    node_x, active = TableHopEncoder(row_embedding)(
        {"table": [x]},
        y,
        graph=_graph({"table": 0}, torch.tensor([0, 1], device=device)),
    )

    assert node_x.device == device
    assert node_x.dtype == dtype
    assert active.dtype == torch.bool
    node_x.square().sum().backward()
    assert x.grad is not None
    assert y.grad is not None
    assert row_embedding.lin.weight.grad is not None
