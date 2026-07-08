# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modified for the structured-data-models package.

import torch
from torch import Tensor
from torch.nn import Parameter


class RMSNorm(torch.nn.Module):
    """Apply root-mean-square normalization over the final dimension.

    The normalization and learned scale are evaluated in ``torch.float32``
    before the result is cast back to the input dtype. This matches the
    released TabFM PyTorch implementation.

    Args:
        channels: Size of the final input dimension.
        epsilon: Non-negative value added before the reciprocal square root.
        device: Device on which to create the learnable scale.
        dtype: Dtype of the learnable scale.
    """

    def __init__(
        self,
        channels: int,
        epsilon: float = 1e-6,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        if epsilon < 0:
            raise ValueError("epsilon must be non-negative")

        self.epsilon = epsilon
        self.weight = Parameter(
            torch.ones(channels, device=device, dtype=dtype)
        )

    def forward(self, input: Tensor) -> Tensor:
        """Normalize ``input`` over its final dimension.

        Args:
            input: Input tensor with shape ``[..., C]``, where ``C`` is
                ``channels``.

        Returns:
            Tensor with shape ``[..., C]`` and the input dtype.
        """
        input_dtype = input.dtype
        input_float = input.float()
        variance = input_float.square().mean(dim=-1, keepdim=True)
        normalized = input_float * (variance + self.epsilon).rsqrt()
        return (normalized * self.weight.float()).to(input_dtype)
