# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import Tensor

from sdm.prior import HP
from sdm.prior.functions import RandomFunction


class RandomTree(RandomFunction):
    """Average the outputs of random symmetric trees.

    Args:
        num_trees: Number of trees in the ensemble.
        depth: Depth of each symmetric tree.
    """

    def __init__(
        self,
        num_trees: HP[int] | int,
        depth: HP[int] | int,
    ) -> None:
        self.num_trees = HP.from_value(num_trees)
        self.depth = HP.from_value(depth)

    def forward(  # noqa: D102
        self,
        x: Tensor,
        out_channels: int,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        num_trees = self.num_trees.sample(generator=generator)
        depth = self.depth.sample(generator=generator)

        x = x.unsqueeze(0)
        batch_size, num_rows, in_channels = x.size()

        prob = x.std(dim=-2, correction=0)
        prob = prob.nan_to_num(0.0, posinf=0.0, neginf=0.0)
        prob = prob.clamp(min=torch.finfo(x.dtype).eps)
        split_col_index = torch.multinomial(
            prob,
            num_samples=num_trees * depth,
            replacement=True,
            generator=generator,
        ).view(batch_size, num_trees, depth)
        split_row_index = torch.randint(
            num_rows,
            (batch_size, num_trees, depth),
            device=x.device,
            generator=generator,
        )
        flat_index = split_row_index * in_channels + split_col_index
        split_value = x.flatten(1).gather(dim=1, index=flat_index.flatten(1))
        split_value = split_value.view(batch_size, num_trees, depth)

        leaf_index = torch.zeros(  # Route all rows through every tree:
            (batch_size, num_trees, num_rows),
            dtype=torch.long,
            device=x.device,
        )
        x = x.transpose(1, 2)  # [batch, in_channels, num_rows]
        for i in range(depth):
            split_col_i = split_col_index[..., i].unsqueeze(-1)
            split_value_i = split_value[..., i].unsqueeze(-1)
            value_i = x.gather(
                dim=1,
                index=split_col_i.expand(-1, -1, num_rows),
            )
            leaf_index *= 2
            leaf_index += value_i > split_value_i

        leaf = torch.randn(
            (batch_size, num_trees, out_channels, 2**depth),
            dtype=x.dtype,
            device=x.device,
            generator=generator,
        )
        leaf = leaf.gather(
            dim=-1,
            index=leaf_index.unsqueeze(-2).expand(-1, -1, out_channels, -1),
        )  # [batch, num_trees, channels, num_rows]

        return leaf.mean(dim=1).transpose(1, 2).squeeze(0)
