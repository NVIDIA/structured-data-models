# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from sdm import CategoricalTensor, TableTensor
from sdm.prior.tabular.transform import PriorTransform


class ShuffleClasses(PriorTransform):
    """Permute the classes of a categorical target."""

    def __call__(  # noqa: D102
        self,
        x: TableTensor,
        y: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> tuple[TableTensor, TableTensor]:
        if y.categorical.size(-1) == 0:
            return x, y
        code = y.categorical.code
        perm = torch.randperm(
            int(code.max()) + 1, device=code.device, generator=generator
        )
        return x, y.replace_blocks(
            categorical=CategoricalTensor(
                code=perm[code], categories=y.categorical.categories
            )
        )
