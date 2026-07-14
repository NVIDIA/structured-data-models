import torch
from sdm.models.kumorfm.invariant_gnn import InvariantGNN
from sdm.testing import withCUDA


@withCUDA
def test_invariant_gnn(device: torch.device) -> None:
    x_dict = {
        "users": torch.randn(4, 8, device=device),
        "orders": torch.randn(8, 8, device=device),
    }
    edge_index_dict = {
        ("orders", "to", "users"): torch.tensor(
            [[0, 1, 2, 3, 4, 5, 6, 7], [0, 0, 1, 1, 2, 2, 3, 3]],
            device=device,
        )
    }

    model = InvariantGNN(channels=8, device=device)
    out = model(
        x_dict=x_dict,
        edge_index_dict=edge_index_dict,
        readout_table="users",
        num_hops=2,
    )
    assert out.size() == (4, 8)
    assert out.device == device
    assert not out.isnan().any()


@withCUDA
def test_invariant_gnn_half_zero_variance_std(device: torch.device) -> None:
    # Since both 'orders' rows are identical, every node receives either
    # identical incoming messages or no messages at all ('users[1]'), such
    # that the standard deviation of every segment is exactly zero:
    x_dict = {
        "users": torch.randn(2, 8, device=device, dtype=torch.half),
        "orders": torch.randn(1, 8, device=device, dtype=torch.half).repeat(
            2, 1
        ),
    }
    edge_index_dict = {
        ("orders", "to", "users"): torch.tensor(
            [[0, 1], [0, 0]],
            device=device,
        )
    }

    model = InvariantGNN(channels=8, device=device, dtype=torch.half)

    std_inputs: list[torch.Tensor] = []
    handle = model.std_lin.register_forward_pre_hook(
        lambda module, args: std_inputs.append(args[0])
    )
    out = model(
        x_dict=x_dict,
        edge_index_dict=edge_index_dict,
        readout_table="users",
        num_hops=2,
    )
    handle.remove()

    assert out.size() == (2, 8)
    assert not out.isnan().any()
    assert len(std_inputs) == 2
    for h in std_inputs:  # One captured input per hop:
        assert (h == 0.0).all()
