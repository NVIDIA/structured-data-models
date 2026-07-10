from types import MappingProxyType

import pytest
import torch
import torch.nn.functional as F
from sdm.nn import InvariantGNN
from sdm.testing import withCUDA


def _generator(
    seed: int = 12_345,
    device: torch.device | str = "cpu",
) -> torch.Generator:
    return torch.Generator(device=device).manual_seed(seed)


def _set_identity(linear: torch.nn.Linear) -> None:
    with torch.no_grad():
        linear.weight.copy_(
            torch.eye(
                linear.weight.size(0),
                device=linear.weight.device,
                dtype=linear.weight.dtype,
            )
        )
        if linear.bias is not None:
            linear.bias.zero_()


def _set_zero(linear: torch.nn.Linear) -> None:
    with torch.no_grad():
        linear.weight.zero_()
        if linear.bias is not None:
            linear.bias.zero_()


def _configure_reducers(
    module: InvariantGNN,
    active: set[str],
) -> None:
    _set_identity(module.src_lin)
    _set_zero(module.edge_type_lin)
    _set_zero(module.skip_lin)
    for name in ("sum", "avg", "std", "min", "max"):
        linear = getattr(module, f"{name}_lin")
        if name in active:
            _set_identity(linear)
        else:
            _set_zero(linear)


def _post_hop(module: InvariantGNN, x: torch.Tensor) -> torch.Tensor:
    return module.post_norm(module.post_lin(module.act(module.norm(x))))


@withCUDA
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_invariant_gnn_aggregates_sampled_edges_and_reads_out_table(
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    module = InvariantGNN(channels=2, device=device, dtype=dtype)
    _configure_reducers(module, {"sum", "avg", "std", "min", "max"})
    relation = ("orders", "placed_by", "users")
    x_dict = MappingProxyType(
        {
            "orders": torch.tensor(
                [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                device=device,
                dtype=dtype,
            ),
            "users": torch.zeros(2, 2, device=device, dtype=dtype),
        }
    )
    edge_index_dict = MappingProxyType(
        {relation: torch.tensor([[0, 1, 2], [0, 0, 1]], device=device)}
    )
    num_sampled_edges_dict = MappingProxyType({relation: [2, 1]})

    out = module(
        x_dict=x_dict,
        edge_index_dict=edge_index_dict,
        num_sampled_edges_dict=num_sampled_edges_dict,
        readout_table="users",
        num_hops=1,
        generator=_generator(device=device),
    )

    reduced = torch.tensor(
        [[11.0, 16.0], [0.0, 0.0]],
        device=device,
        dtype=dtype,
    )
    torch.testing.assert_close(out, _post_hop(module, reduced))
    assert out.shape == x_dict["users"].shape
    assert out.isfinite().all()


def test_invariant_gnn_uses_generator_deterministically() -> None:
    channels = 3
    module = InvariantGNN(channels=channels)
    _set_zero(module.src_lin)
    _set_identity(module.edge_type_lin)
    _set_zero(module.skip_lin)
    _set_identity(module.sum_lin)
    for linear in (
        module.avg_lin,
        module.std_lin,
        module.min_lin,
        module.max_lin,
    ):
        _set_zero(linear)

    relation = ("orders", "placed_by", "users")
    x_dict = {
        "orders": torch.zeros(1, channels),
        "users": torch.zeros(1, channels),
    }
    edge_index_dict = {relation: torch.tensor([[0], [0]])}
    num_sampled_edges_dict = {relation: [1]}

    edge_type_emb = torch.randn(2, channels, generator=_generator())
    edge_type_emb = F.normalize(edge_type_emb, dim=-1)
    expected = _post_hop(module, edge_type_emb[:1])

    kwargs = {
        "x_dict": x_dict,
        "edge_index_dict": edge_index_dict,
        "num_sampled_edges_dict": num_sampled_edges_dict,
        "readout_table": "users",
        "num_hops": 1,
    }
    out = module(**kwargs, generator=_generator())
    repeated = module(**kwargs, generator=_generator())
    different = module(**kwargs, generator=_generator(seed=54_321))

    torch.testing.assert_close(out, expected)
    torch.testing.assert_close(repeated, out)
    assert not torch.equal(different, out)


def test_invariant_gnn_zero_hops_returns_readout_without_randomness() -> None:
    module = InvariantGNN(channels=3)
    users = torch.randn(2, 3)
    generator = _generator()
    generator_state = generator.get_state()

    out = module(
        x_dict={"users": users},
        edge_index_dict={},
        num_sampled_edges_dict={},
        readout_table="users",
        num_hops=0,
        generator=generator,
    )

    assert out is users
    torch.testing.assert_close(generator.get_state(), generator_state)


def test_to_bidirectional_completes_and_coalesces_relations() -> None:
    edge_index_dict = {
        ("A", "to", "B"): torch.tensor([[0, 1], [10, 11]]),
        ("B", "to", "C"): torch.tensor([[10, 11], [100, 101]]),
        ("C", "rev_to", "B"): torch.tensor([[100, 102], [10, 12]]),
    }

    out = InvariantGNN.to_bidirectional(edge_index_dict)

    assert list(out) == [
        ("A", "to", "B"),
        ("B", "rev_to", "A"),
        ("B", "to", "C"),
        ("C", "rev_to", "B"),
    ]
    torch.testing.assert_close(
        out[("A", "to", "B")],
        torch.tensor([[0, 1], [10, 11]]),
    )
    torch.testing.assert_close(
        out[("B", "rev_to", "A")],
        torch.tensor([[10, 11], [0, 1]]),
    )
    torch.testing.assert_close(
        out[("B", "to", "C")],
        torch.tensor([[10, 11, 12], [100, 101, 102]]),
    )
    torch.testing.assert_close(
        out[("C", "rev_to", "B")],
        torch.tensor([[100, 101, 102], [10, 11, 12]]),
    )


def test_invariant_gnn_chunked_recurrent_gradients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("sdm.nn.invariant_gnn._DST_CHUNK_SIZE", 2)
    channels = 2
    num_nodes = 3
    module = InvariantGNN(channels=channels)
    x = torch.randn(num_nodes, channels, requires_grad=True)
    relation = ("node", "next", "node")
    edge_index = torch.stack(
        (
            torch.arange(num_nodes - 1),
            torch.arange(1, num_nodes),
        )
    )

    out = module(
        x_dict={"node": x},
        edge_index_dict={relation: edge_index},
        num_sampled_edges_dict={relation: [num_nodes - 1, 0]},
        readout_table="node",
        num_hops=2,
        generator=_generator(),
    )
    out.square().mean().backward()

    assert out.shape == x.shape
    assert out.isfinite().all()
    assert x.grad is not None
    assert x.grad.isfinite().all()
    for parameter in module.parameters():
        assert parameter.grad is not None
        assert parameter.grad.isfinite().all()
