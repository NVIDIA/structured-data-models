import pytest
import torch
import torch.nn.functional as F
from sdm.nn import InvariantGNN
from sdm.testing import withCUDA


def _generator(device: torch.device | str = "cpu") -> torch.Generator:
    return torch.Generator(device=device).manual_seed(12345)


def _set_identity(linear: torch.nn.Linear) -> None:
    with torch.no_grad():
        linear.weight.copy_(torch.eye(linear.weight.size(0)))
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


def test_invariant_gnn_reducers_and_empty_neighborhoods() -> None:
    module = InvariantGNN(channels=2, dst_chunk_size=None)
    _configure_reducers(module, {"sum", "avg", "std", "min", "max"})
    x = torch.tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
        ],
    )
    edge_index = torch.tensor(
        [
            [2, 0, 1],
            [1, 0, 0],
        ],
    )
    edge_type = torch.zeros(3, dtype=torch.long)

    out = module(
        x=x,
        edge_index=edge_index,
        edge_type=edge_type,
        num_edge_types=1,
        generator=_generator(),
    )

    reduced = torch.tensor(
        [
            [11.0, 16.0],
            [20.0, 24.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ],
    )
    torch.testing.assert_close(out, _post_hop(module, reduced))
    assert out.isfinite().all()


def test_invariant_gnn_std_epsilon_behavior() -> None:
    module = InvariantGNN(channels=2, dst_chunk_size=None)
    _configure_reducers(module, {"std"})
    x = torch.tensor(
        [
            [0.0, 0.0],
            [0.004, 0.008],
            [0.0, 0.0],
        ],
    )
    edge_index = torch.tensor([[0, 1], [2, 2]])
    edge_type = torch.zeros(2, dtype=torch.long)

    out = module(
        x=x,
        edge_index=edge_index,
        edge_type=edge_type,
        num_edge_types=1,
        generator=_generator(),
    )

    reduced = torch.tensor(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.004],
        ],
    )
    torch.testing.assert_close(out, _post_hop(module, reduced))


def test_invariant_gnn_seeded_normalized_edge_embeddings() -> None:
    channels = 3
    module = InvariantGNN(channels=channels, dst_chunk_size=None)
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

    x = torch.zeros(3, channels)
    edge_index = torch.tensor([[0, 1], [2, 1]])
    edge_type = torch.tensor([1, 0])

    edge_type_emb = torch.randn(2, channels, generator=_generator())
    edge_type_emb = F.normalize(edge_type_emb, dim=-1)
    reduced = torch.zeros_like(x)
    reduced[2] = edge_type_emb[1]
    reduced[1] = edge_type_emb[0]
    expected = _post_hop(module, reduced)

    out = module(
        x=x,
        edge_index=edge_index,
        edge_type=edge_type,
        num_edge_types=2,
        generator=_generator(),
    )
    repeated = module(
        x=x,
        edge_index=edge_index,
        edge_type=edge_type,
        num_edge_types=2,
        generator=_generator(),
    )
    different = module(
        x=x,
        edge_index=edge_index,
        edge_type=edge_type,
        num_edge_types=2,
        generator=torch.Generator().manual_seed(54321),
    )

    torch.testing.assert_close(edge_type_emb.norm(dim=-1), torch.ones(2))
    torch.testing.assert_close(out, expected)
    torch.testing.assert_close(repeated, out)
    assert not torch.equal(different, out)


@withCUDA
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_invariant_gnn_without_edges_preserves_dtype_and_device(
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    module = InvariantGNN(
        channels=4,
        dst_chunk_size=None,
        device=device,
        dtype=dtype,
    )
    x = torch.randn(5, 4, device=device, dtype=dtype, requires_grad=True)
    edge_index = torch.empty(2, 0, dtype=torch.long, device=device)
    edge_type = torch.empty(0, dtype=torch.long, device=device)

    out = module(
        x=x,
        edge_index=edge_index,
        edge_type=edge_type,
        num_edge_types=0,
    )
    expected = _post_hop(module, module.skip_lin(x))

    torch.testing.assert_close(out, expected)
    assert out.shape == x.shape
    assert out.dtype == dtype
    assert out.device == device
    assert out.isfinite().all()

    out.square().sum().backward()
    assert x.grad is not None
    assert module.skip_lin.weight.grad is not None
    assert module.post_lin.weight.grad is not None


def test_invariant_gnn_composes_with_cpu_autocast() -> None:
    module = InvariantGNN(channels=4, dst_chunk_size=None)
    x = torch.randn(5, 4, requires_grad=True)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
    edge_type = torch.tensor([0, 1, 0, 1])

    with torch.autocast("cpu", dtype=torch.bfloat16):
        encoded = torch.nn.functional.linear(
            x,
            torch.eye(4),
        )
        out = module(
            x=encoded,
            edge_index=edge_index,
            edge_type=edge_type,
            num_edge_types=2,
            generator=_generator(),
        )

    assert encoded.dtype == torch.bfloat16
    assert out.dtype == torch.bfloat16
    out.float().square().sum().backward()
    assert x.grad is not None
    assert module.src_lin.weight.grad is not None


def test_invariant_gnn_two_hops() -> None:
    module = InvariantGNN(channels=4, dst_chunk_size=None)
    x = torch.randn(5, 4, requires_grad=True)
    edge_index = torch.tensor(
        [
            [0, 1, 1, 2, 2, 3, 3, 4],
            [1, 0, 2, 1, 3, 2, 4, 3],
        ],
    )
    edge_type = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])

    one_hop = module(
        x=x,
        edge_index=edge_index,
        edge_type=edge_type,
        num_edge_types=2,
        generator=_generator(),
    )
    two_hops = module(
        x=x,
        edge_index=edge_index,
        edge_type=edge_type,
        num_edge_types=2,
        num_hops=2,
        generator=_generator(),
    )

    assert two_hops.shape == x.shape
    assert two_hops.isfinite().all()
    assert not torch.equal(two_hops, one_hop)
    two_hops.square().sum().backward()
    assert x.grad is not None
    assert module.src_lin.weight.grad is not None


def test_invariant_gnn_zero_hops_is_identity() -> None:
    module = InvariantGNN(channels=4)
    x = torch.randn(5, 4)
    edge_index = torch.tensor([[0, 1], [1, 2]])
    edge_type = torch.tensor([0, 0])

    out = module(
        x=x,
        edge_index=edge_index,
        edge_type=edge_type,
        num_edge_types=1,
        num_hops=0,
    )

    assert out is x


def test_invariant_gnn_chunked_output_and_gradients() -> None:
    full_module = InvariantGNN(channels=4, dst_chunk_size=None)
    chunked_module = InvariantGNN(channels=4, dst_chunk_size=2)
    chunked_module.load_state_dict(full_module.state_dict())

    x = torch.randn(8, 4)
    full_x = x.clone().requires_grad_()
    chunked_x = x.clone().requires_grad_()
    edge_index = torch.tensor(
        [
            [4, 1, 7, 3, 2, 6, 0, 5, 2, 0],
            [3, 0, 5, 1, 6, 4, 3, 3, 0, 5],
        ],
    )
    edge_type = torch.tensor([0, 1, 2, 0, 1, 2, 1, 0, 2, 1])

    full_out = full_module(
        x=full_x,
        edge_index=edge_index,
        edge_type=edge_type,
        num_edge_types=3,
        num_hops=2,
        generator=_generator(),
    )
    chunked_out = chunked_module(
        x=chunked_x,
        edge_index=edge_index,
        edge_type=edge_type,
        num_edge_types=3,
        num_hops=2,
        generator=_generator(),
    )
    torch.testing.assert_close(chunked_out, full_out)

    full_out.square().sum().backward()
    chunked_out.square().sum().backward()
    assert full_x.grad is not None
    assert chunked_x.grad is not None
    torch.testing.assert_close(chunked_x.grad, full_x.grad)

    full_parameters = dict(full_module.named_parameters())
    chunked_parameters = dict(chunked_module.named_parameters())
    assert full_parameters.keys() == chunked_parameters.keys()
    for name, full_parameter in full_parameters.items():
        chunked_parameter = chunked_parameters[name]
        assert full_parameter.grad is not None
        assert chunked_parameter.grad is not None
        torch.testing.assert_close(
            chunked_parameter.grad,
            full_parameter.grad,
        )


def test_invariant_gnn_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="`channels` must be positive"):
        InvariantGNN(channels=0)
    with pytest.raises(ValueError, match="`dst_chunk_size` must be positive"):
        InvariantGNN(channels=2, dst_chunk_size=0)


def test_invariant_gnn_rejects_malformed_inputs() -> None:
    module = InvariantGNN(channels=3)
    x = torch.randn(4, 3)
    edge_index = torch.tensor([[0, 1], [1, 2]])
    edge_type = torch.tensor([0, 1])

    with pytest.raises(ValueError, match="`num_hops` must be non-negative"):
        module(x, edge_index, edge_type, num_edge_types=2, num_hops=-1)
    with pytest.raises(
        ValueError,
        match="`num_edge_types` must be non-negative",
    ):
        module(x, edge_index, edge_type, num_edge_types=-1)
    with pytest.raises(ValueError, match="`x` must have shape"):
        module(x[:, :2], edge_index, edge_type, num_edge_types=2)
    with pytest.raises(ValueError, match="floating-point dtype"):
        module(x.long(), edge_index, edge_type, num_edge_types=2)
    with pytest.raises(ValueError, match="same dtype as the module"):
        InvariantGNN(channels=3, dtype=torch.float64)(
            x,
            edge_index,
            edge_type,
            num_edge_types=2,
        )
    with pytest.raises(ValueError, match="`edge_index` must have shape"):
        module(x, edge_index[:1], edge_type, num_edge_types=2)
    with pytest.raises(ValueError, match="`edge_index` must have dtype"):
        module(x, edge_index.int(), edge_type, num_edge_types=2)
    with pytest.raises(ValueError, match="`edge_type` must have shape"):
        module(x, edge_index, edge_type[:1], num_edge_types=2)
    with pytest.raises(ValueError, match="`edge_type` must have dtype"):
        module(x, edge_index, edge_type.int(), num_edge_types=2)
    with pytest.raises(ValueError, match="out-of-range node ID"):
        module(
            x,
            torch.tensor([[0, 4], [1, 2]]),
            edge_type,
            num_edge_types=2,
        )
    with pytest.raises(ValueError, match="out-of-range edge-type ID"):
        module(x, edge_index, torch.tensor([0, 2]), num_edge_types=2)

    meta_module = InvariantGNN(channels=3, device=torch.device("meta"))
    with pytest.raises(ValueError, match="same device as the module"):
        meta_module(x, edge_index, edge_type, num_edge_types=2)
    with pytest.raises(ValueError, match="same device as `x`"):
        module(
            x,
            edge_index.to(device="meta"),
            edge_type,
            num_edge_types=2,
        )
