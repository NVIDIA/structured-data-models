from textwrap import dedent

import torch
from sdm import RelationalData


def test_repr(data: RelationalData) -> None:
    assert repr(data) == dedent("""\
        RelationalData(
          tables={
            users: TableTensor(
              size=(4, 3),
              blocks={
                numerical (1): [age],
                categorical (1): [city],
                id (1): [user_id],
              },
            ),
            orders: TableTensor(
              size=(6, 3),
              blocks={
                numerical (1): [amount],
                id (2): [user_id, item_id],
              },
            ),
            items: TableTensor(
              size=(3, 2),
              blocks={
                categorical (1): [category],
                id (1): [item_id],
              },
            ),
          },
          relationships=[
            orders.user_id<>users.user_id,
            orders.item_id<>items.item_id,
          ],
        )""")


def test_repr(data: RelationalData) -> None:
    assert repr(data) == dedent("""\
        RelationalData(
          tables={
            users: TableTensor(
              size=(4, 3),
              blocks={
                numerical (1): [age],
                categorical (1): [city],
                id (1): [user_id],
              },
            ),
            orders: TableTensor(
              size=(6, 3),
              blocks={
                numerical (1): [amount],
                id (2): [user_id, item_id],
              },
            ),
            items: TableTensor(
              size=(3, 2),
              blocks={
                categorical (1): [category],
                id (1): [item_id],
              },
            ),
          },
          relationships=[
            orders.user_id<>users.user_id,
            orders.item_id<>items.item_id,
          ],
        )""")


def test_edge_indices(data: RelationalData) -> None:
    edge_indices = data.edge_indices()

    assert len(edge_indices) == 2
    assert edge_indices[0].equal(
        torch.tensor([[0, 1, 2, 3, 4, 5], [0, 0, 1, 3, 3, 3]])
    )
    assert edge_indices[1].equal(
        torch.tensor([[0, 1, 2, 3, 4, 5], [0, 1, 2, 0, 1, 0]])
    )
