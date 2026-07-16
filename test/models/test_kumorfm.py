import torch
from sdm import ColumnarTensor, Stype, TableTensor
from sdm.models import KumoRFM
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
    # Every segment sees identical or no incoming messages, so std is zero:
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


def test_default_recipe_preserves_ids() -> None:
    table = TableTensor(
        columns={
            Stype.numerical: ("value",),
            Stype.id: ("entity_id",),
        },
        numerical=torch.tensor([[1.0], [2.0]]),
        id=ColumnarTensor((torch.tensor([10, 11]),)),
    )

    transformed = KumoRFM.default_recipe().features.fit_transform(table)

    assert transformed.columns[Stype.id] == ("entity_id",)
    assert transformed.id is table.id
