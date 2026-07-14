import torch
from sdm import RelatedTables, RelationalSample, TableTensor


def test_sample_preserves_exact_edges_and_same_device() -> None:
    edge_index = torch.tensor([[1], [0]])
    task_edge_index = torch.tensor([[0], [0]])
    sample = RelationalSample(
        node_batch={"items": torch.tensor([0, 0])},
        node_hops={"items": torch.tensor([0, 1])},
        edge_indices=(edge_index,),
        edge_hops=(torch.tensor([1]),),
        task_edge_indices=(task_edge_index,),
        num_hops=1,
        num_neighbors=(1,),
        disjoint=True,
        temporal=False,
        temporal_strategy="uniform",
    )
    related_tables = RelatedTables(
        tables={"items": TableTensor.from_tensor(torch.ones(2, 1), ["id"])},
        relationships=(
            {
                "left_table": "items",
                "left_column": "id",
                "right_table": "items",
                "right_column": "id",
            },
        ),
        task_links=(
            {
                "task_column": "id",
                "table": "items",
                "table_column": "id",
            },
        ),
        sample=sample,
    )

    assert sample.to("cpu") is sample
    edges, task_edges = related_tables.edge_indices(
        TableTensor.from_tensor(torch.ones(1, 1), ["id"])
    )
    assert edges[0] is edge_index
    assert task_edges[0] is task_edge_index
