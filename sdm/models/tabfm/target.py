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

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear


class OneHotAndLinear(torch.nn.Module):
    """Project one-hot TabFM classification targets into hidden channels.

    Class IDs outside ``[0, num_classes)`` map to an all-zero one-hot vector,
    so their output is the projection bias. This matches the released TabFM
    behavior for padded or otherwise invalid targets and preserves the
    ``projection`` checkpoint parameter names.

    Args:
        num_classes: Number of valid class IDs.
        channels: Number of output representation channels.
        device: Device on which to create parameters.
        dtype: Dtype of the parameters.
    """

    def __init__(
        self,
        num_classes: int,
        channels: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        if channels <= 0:
            raise ValueError("channels must be positive")

        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}
        self.num_classes = num_classes
        self.projection = Linear(
            num_classes,
            channels,
            **factory_kwargs,
        )

    def forward(self, target: Tensor) -> Tensor:
        """Encode integer class targets.

        Args:
            target: Class targets with shape ``[..., T]``.

        Returns:
            Target representations with shape ``[..., T, D]``, where ``D``
            is ``channels``.
        """
        target = target.long()
        valid = (target >= 0) & (target < self.num_classes)
        invalid_class = target.new_full((), self.num_classes)
        mapped = torch.where(valid, target, invalid_class)
        one_hot = F.one_hot(mapped, self.num_classes + 1)
        one_hot = one_hot[..., : self.num_classes]
        return self.projection(one_hot.to(self.projection.weight.dtype))
