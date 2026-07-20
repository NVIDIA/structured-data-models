import torch
from sdm import ColumnarTensor, RelatedTables, Stype, TableTensor
from sdm.models import KumoRFM
from sdm.models.kumorfm.graph import HomogeneousGraph
from sdm.models.kumorfm.invariant_gnn import InvariantGNN
from sdm.testing import withCUDA


@withCUDA
def test_invariant_gnn(device: torch.device) -> None:
    related_tables = RelatedTables(
        tables={
            "users": TableTensor(
                columns={"numerical": ("num",), "id": ("id",)},
                numerical=torch.randn(4, 1, device=device),
                id=ColumnarTensor(
                    (torch.tensor([0, 1, 2, 3], device=device),)
                ),
            ),
            "orders": TableTensor(
                columns={"numerical": ("num",), "id": ("id",)},
                numerical=torch.randn(8, 1, device=device),
                id=ColumnarTensor(
                    (torch.tensor([0, 0, 1, 1, 2, 2, 3, 3], device=device),)
                ),
            ),
        },
        relationships=[
            {
                "left_table": "orders",
                "left_column": "id",
                "right_table": "users",
                "right_column": "id",
            }
        ],
        task_links=[],
    )

    graph = HomogeneousGraph.from_related_tables(related_tables)
    assert graph.colptr.equal(
        torch.tensor(
            [0, 2, 4, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16], device=device
        )
    )
    col = torch.repeat_interleave(
        torch.arange(graph.colptr.numel() - 1, device=graph.colptr.device),
        graph.colptr.diff(),
    )
    # Make `row` deterministic within local neighborhoods:
    row, perm = graph.row.sort()
    row = row[col[perm].argsort(stable=True)]
    assert row.equal(
        torch.tensor(
            [4, 5, 6, 7, 8, 9, 10, 11, 0, 0, 1, 1, 2, 2, 3, 3], device=device
        )
    )
    assert graph.edge_type.equal(
        torch.tensor(
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1], device=device
        )
    )
    assert graph.num_edge_types == 2
    assert graph.start_node_offsets == {"users": 0, "orders": 4}
    assert graph.end_node_offsets == {"users": 4, "orders": 12}

    model = InvariantGNN(channels=8, device=device)
    out = model(
        x=torch.randn(12, 8, device=device),
        graph=graph,
        readout_table="users",
        num_hops=2,
    )
    assert out.size() == (4, 8)
    assert out.device == device
    assert not out.isnan().any()


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
