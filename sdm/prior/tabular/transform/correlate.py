# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import Tensor

from sdm import CategoricalTensor, TableTensor
from sdm.prior import HP, Choice, Uniform
from sdm.prior.tabular.transform import PriorTransform
from sdm.prior.tabular.transform._table import (
    block_order,
    categorical_mask,
    feature_values,
)


def _discretize_correlated_column(
    column: Tensor,
    latent: Tensor,
    rho: Tensor,
    generator: torch.Generator | None,
) -> Tensor:
    r"""Blend a categorical column in latent space and restore its levels.

    Randomly rank nominal levels, mix with the shared latent, then map score
    quantiles back to the input levels and counts.
    """
    values, inverse, counts = torch.unique(
        column,
        sorted=True,
        return_inverse=True,
        return_counts=True,
    )
    num_categories = values.numel()
    if num_categories < 2:
        return column

    category_order = torch.randperm(
        num_categories,
        device=column.device,
        generator=generator,
    )
    category_rank = torch.empty_like(category_order)
    category_rank[category_order] = torch.arange(
        num_categories,
        device=column.device,
    )
    original_score = category_rank[inverse].to(column.dtype)
    original_score = original_score - original_score.mean()
    scale = original_score.std(correction=0).clamp(min=1e-6)
    original_score = original_score / scale

    independent_scale = (1 - rho.square()).clamp(min=0).sqrt()
    score = rho * latent + independent_scale * original_score
    row_order = torch.argsort(score, stable=True)
    ordered_values = torch.repeat_interleave(
        input=values[category_order],
        repeats=counts[category_order],
    )
    out = torch.empty_like(column)
    out[row_order] = ordered_values
    return out


class CorrelateColumns(PriorTransform):
    """Blend random feature blocks toward shared Gaussian factors.

    Categorical columns retain their values, counts, and missing cells.

    Args:
        num_blocks: Number of latent blocks per dataset.
            Defaults to ``Choice(1, 2, 3, 4)``.
        rho: Correlation strength per block. Defaults to
            ``Uniform(0.2, 0.7)``.
    """

    def __init__(
        self,
        *,
        num_blocks: HP[int] | int | None = None,
        rho: HP[float] | float | None = None,
    ) -> None:
        self.num_blocks = HP.from_value(
            Choice(1, 2, 3, 4) if num_blocks is None else num_blocks
        )
        self.rho = HP.from_value(Uniform(0.2, 0.7) if rho is None else rho)

    def __call__(  # noqa: D102
        self,
        x: TableTensor,
        y: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> tuple[TableTensor, TableTensor]:
        values = feature_values(x)
        num_columns = values.size(-1)
        if num_columns < 2:
            return x, y

        num_numerical = x.numerical.size(-1)
        cat_mask = categorical_mask(x)
        output = values.clone()
        k = int(
            torch.randint(
                low=2,
                high=num_columns + 1,
                size=(),
                device=x.device,
                generator=generator,
            )
        )
        columns = torch.randperm(
            num_columns,
            device=x.device,
            generator=generator,
        )[:k]
        num_blocks = self.num_blocks.sample(generator=generator)
        block_ids = torch.randint(
            high=num_blocks,
            size=(k,),
            device=x.device,
            generator=generator,
        )
        for block_id in range(num_blocks):
            block = columns[block_ids == block_id]
            if block.numel() >= 2:
                rho = (
                    output.new_empty(()).uniform_(
                        self.rho.low, self.rho.high, generator=generator
                    )
                    if isinstance(self.rho, Uniform)
                    else output.new_tensor(
                        self.rho.sample(generator=generator)
                    )
                )
                latent = torch.randn(
                    size=(output.size(-2), 1),
                    device=output.device,
                    dtype=output.dtype,
                    generator=generator,
                )
                categorical = cat_mask[block]
                numerical_columns = block[~categorical]
                if numerical_columns.numel() > 0:
                    output[:, numerical_columns] = (1 - rho) * output[
                        :, numerical_columns
                    ] + rho * latent
                for column in block[categorical].tolist():
                    observed = ~output[:, column].isnan()
                    output[observed, column] = _discretize_correlated_column(
                        column=output[observed, column],
                        latent=latent[observed, 0],
                        rho=rho,
                        generator=generator,
                    )

        order = block_order(x)
        output = output[:, order]
        values = values[:, order]
        if not x.categorical.size(-1):
            return x.replace_blocks(numerical=output), y
        order = values[:, num_numerical:].argsort(dim=0, stable=True)
        sorted_values = values[:, num_numerical:].gather(0, order)
        indices = torch.searchsorted(
            sorted_sequence=sorted_values.mT.contiguous(),
            input=output[:, num_numerical:].mT.contiguous(),
        ).mT.clamp(max=x.size(-2) - 1)
        code = x.categorical.code.gather(0, order).gather(0, indices)
        code = code.masked_fill(output[:, num_numerical:].isnan(), -1)
        return x.replace_blocks(
            numerical=output[:, :num_numerical],
            categorical=CategoricalTensor(
                code=code, categories=x.categorical.categories
            ),
        ), y
