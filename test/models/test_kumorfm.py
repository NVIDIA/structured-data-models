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
