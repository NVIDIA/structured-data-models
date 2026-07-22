import torch
from sdm import RelatedTables, TableTensor


def test_related_tables_with_tables() -> None:
    users = TableTensor(
        columns={"numerical": ("age",)},
        numerical=torch.tensor([[42.0], [23.0], [31.0]]),
    )
    orders = TableTensor(
        columns={"numerical": ("amount",)},
        numerical=torch.tensor([[9.99], [31.57], [29.97], [19.49]]),
    )
    related_tables = RelatedTables(
        tables={"users": users, "orders": orders},
        relationships=[
            {
                "left_table": "orders",
                "left_columns": "user_id",
                "right_table": "users",
                "right_columns": "user_id",
            }
        ],
        task_links=[
            {
                "task_columns": "user_id",
                "table": "users",
                "table_columns": "user_id",
            }
        ],
    )

    out = related_tables.with_tables(
        {
            "users": users[:2],
            "orders": orders[:3],
        }
    )

    assert out.tables["users"].equal(users[:2])
    assert out.tables["orders"].equal(orders[:3])
    assert out.relationships == related_tables.relationships
    assert out.task_links == related_tables.task_links
    assert related_tables.tables["users"].equal(users)
    assert related_tables.tables["orders"].equal(orders)
