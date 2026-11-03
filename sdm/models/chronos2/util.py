# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence

import torch
from torch import Tensor


def rollout_quantiles(samples: Tensor, levels: Sequence[float]) -> Tensor:
    """Combine the nine forecast paths using weighted empirical CDF.

    Args:
        samples: Predictions with shape ``[S, 9 * Q, H]``, where ``S`` is
            series, ``Q`` is quantiles and ``H`` is forecast steps.
        levels: The model's quantile levels.
    """
    quantiles = torch.tensor(levels, device=samples.device)
    paths = torch.arange(1, 10, device=samples.device).float() / 10

    def mass(q: Tensor) -> Tensor:
        boundaries = torch.cat([q.new_zeros(1), q, q.new_ones(1)])
        weights = (boundaries[2:] - boundaries[:-2]) / 2
        return weights / weights.sum()

    weights = (mass(paths)[:, None] * mass(quantiles)[None, :]).flatten()
    weights = weights / weights.sum()
    values, index = samples.transpose(-1, -2).sort(-1)
    cdf = weights.expand_as(values).gather(-1, index).cumsum(-1).clamp(0, 1)
    cdf, index = cdf.sort(-1)
    values = values.gather(-1, index)
    # The first CDF entry is positive. All model levels are below the final
    # CDF entry, so only the lower endpoint is needed for interpolation.
    cdf = torch.cat([cdf.new_zeros((*cdf.shape[:-1], 1)), cdf], dim=-1)
    values = torch.cat([values[..., :1], values], dim=-1)
    query = quantiles.expand(*cdf.shape[:-1], -1).contiguous()
    upper = torch.searchsorted(cdf.contiguous(), query, right=True)
    upper = upper.clamp(max=cdf.size(-1) - 1)
    lower = upper - 1
    lo, hi = cdf.gather(-1, lower), cdf.gather(-1, upper)
    weight = ((query - lo) / (hi - lo)).nan_to_num(nan=0.0)
    lo, hi = values.gather(-1, lower), values.gather(-1, upper)
    return (lo + weight * (hi - lo)).transpose(-1, -2)
