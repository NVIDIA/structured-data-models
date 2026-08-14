# Copyright 2026 Google LLC
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
#
# Modified for the structured-data-models package.

import math
from typing import Any

import torch

from sdm.nn import RotaryEmbedding, SoftplusScale


def _attention_transforms(
    channels: int,
    num_heads: int,
    rope_theta: float | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> tuple[torch.nn.Sequential, torch.nn.Sequential]:
    """Construct TabFM query and key head transformations."""
    head_channels = channels // num_heads
    factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}
    query: list[torch.nn.Module] = []
    key: list[torch.nn.Module] = []
    if rope_theta is not None:
        query.append(
            RotaryEmbedding(
                channels=head_channels,
                layout="interleaved",
                theta=rope_theta,
                requires_grad=False,
                **factory_kwargs,
            )
        )
        key.append(
            RotaryEmbedding(
                channels=head_channels,
                layout="interleaved",
                theta=rope_theta,
                requires_grad=False,
                **factory_kwargs,
            )
        )
    query.extend(
        [
            torch.nn.RMSNorm(head_channels, eps=1e-6, **factory_kwargs),
            SoftplusScale(
                head_channels,
                multiplier=1.442695041 / math.sqrt(head_channels),
                **factory_kwargs,
            ),
        ]
    )
    key.append(torch.nn.RMSNorm(head_channels, eps=1e-6, **factory_kwargs))
    return torch.nn.Sequential(*query), torch.nn.Sequential(*key)
