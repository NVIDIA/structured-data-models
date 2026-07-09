import pandas as pd
import torch
from sdm import RelatedTables, RelationalData, TableTensor
from sdm.relational.sampler import RelationalSampler, _convert_hetero_sample


def test_related_tables_metadata_defaults_to_none() -> None:
    related_tables = RelatedTables(
        tables={},
        relationships=(),
        task_links=(),
    )

    assert related_tables.metadata is None


def test_sampler_uses_paired_reverse_relation_keys() -> None:
    orders = TableTensor.from_pandas(
        df=pd.DataFrame({"user_id": [1, 2]}),
        stypes={"user_id": "id"},
    )
    users = TableTensor.from_pandas(
        df=pd.DataFrame({"user_id": [1, 2]}),
        stypes={"user_id": "id"},
    )
    data = RelationalData(
        tables={"orders": orders, "users": users},
        relationships=[
            {
                "left_table": "orders",
                "left_column": "user_id",
                "right_table": "users",
                "right_column": "user_id",
            }
        ],
    )

    sampler = RelationalSampler(data)

    assert tuple(sampler._row_dict) == (
        ("orders", "relationship_0", "users"),
        ("users", "rev_relationship_0", "orders"),
    )


def test_convert_hetero_sample() -> None:
    forward = ("orders", "relationship_0", "users")
    reverse = ("users", "rev_relationship_0", "orders")
    forward_key = "__".join(forward)
    reverse_key = "__".join(reverse)
    row_dict: dict[str, torch.Tensor] = {
        forward_key: torch.tensor([0, 2]),
        reverse_key: torch.empty(0, dtype=torch.long),
    }
    col_dict: dict[str, torch.Tensor] = {
        forward_key: torch.tensor([1, 0]),
        reverse_key: torch.empty(0, dtype=torch.long),
    }
    node_dict = {
        "orders": torch.tensor([[0, 12], [0, 13], [1, 14]]),
        "users": torch.tensor([[0, 20], [1, 21]]),
        "unused": torch.empty((0, 2), dtype=torch.long),
    }
    num_sampled_nodes_dict = {
        "orders": [0, 3, 0],
        "users": [2, 0, 0],
        "unused": [0, 0, 0],
    }
    num_sampled_edges_dict: dict[str, list[int]] = {
        forward_key: [2, 0],
        reverse_key: [0, 0],
    }
    seed_time = torch.tensor([100, 200])

    node_index_dict, metadata = _convert_hetero_sample(
        row_dict=row_dict,
        col_dict=col_dict,
        node_dict=node_dict,
        num_sampled_nodes_dict=num_sampled_nodes_dict,
        num_sampled_edges_dict=num_sampled_edges_dict,
        edge_type_by_key={
            forward_key: forward,
            reverse_key: reverse,
        },
        seed_time=seed_time,
    )

    assert node_index_dict["orders"].equal(torch.tensor([12, 13, 14]))
    assert node_index_dict["users"].equal(torch.tensor([20, 21]))
    assert node_index_dict["unused"].numel() == 0
    assert metadata.batch_dict["orders"].equal(torch.tensor([0, 0, 1]))
    assert metadata.batch_dict["users"].equal(torch.tensor([0, 1]))
    assert metadata.batch_dict["unused"].numel() == 0
    assert metadata.edge_index_dict[forward].equal(
        torch.tensor([[0, 2], [1, 0]])
    )
    assert metadata.edge_index_dict[reverse].size() == (2, 0)
    assert metadata.num_sampled_nodes_dict == num_sampled_nodes_dict
    assert metadata.num_sampled_edges_dict == {
        forward: [2, 0],
        reverse: [0, 0],
    }
    assert metadata.seed_time is seed_time
