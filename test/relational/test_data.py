# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from textwrap import dedent

import torch

from sdm import RelationalData
from sdm.testing import withCUDA


def test_repr(relational_data: RelationalData) -> None:
    assert repr(relational_data) == dedent("""\
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
              size=(6, 4),
              blocks={
                numerical (1): [amount],
                datetime (1): [timestamp],
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
            orders.user_id <> users.user_id,
            orders.item_id <> items.item_id,
          ],
        )""")


@withCUDA
def test_edge_indices(
    relational_data: RelationalData,
    device: torch.device,
) -> None:
    edge_indices = relational_data.edge_indices()

    assert len(edge_indices) == 2
    assert edge_indices[0].equal(
        torch.tensor(
            [[0, 1, 2, 3, 4, 5], [0, 0, 1, 3, 3, 3]],
            device=device,
        )
    )
    assert edge_indices[1].equal(
        torch.tensor(
            [[0, 1, 2, 3, 4, 5], [0, 1, 2, 0, 1, 0]],
            device=device,
        )
    )
