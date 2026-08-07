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

from typing import Any, cast

import torch
from torch import Tensor
from torch.nn import Embedding, Linear

from sdm.models.tabfm.mlp import MLP


class TargetEmbedding(torch.nn.Module):
    """Embed context targets without exposing query targets."""

    def __init__(
        self,
        channels: int,
        is_classifier: bool,
        max_classes: int = 10,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}
        self.is_classifier = is_classifier
        self.lookup: Embedding | MLP
        if is_classifier:
            self.lookup = Embedding(max_classes, channels, **factory_kwargs)
        else:
            self.lookup = MLP(
                in_channels=1,
                hidden_channels=(6,),
                out_channels=channels,
                **factory_kwargs,
            )

    def forward(self, targets: Tensor, context_size: Tensor) -> Tensor:
        """Embed targets with shape ``[B, T]`` and zero query rows."""
        row_index = torch.arange(targets.size(-1), device=targets.device)
        context = row_index < context_size.unsqueeze(-1)
        targets = targets.where(context, 0)
        if self.is_classifier:
            assert isinstance(self.lookup, Embedding)
            embedding = self.lookup(
                targets.long().clamp(0, self.lookup.num_embeddings - 1)
            )
        else:
            assert isinstance(self.lookup, MLP)
            weight = cast(Linear, self.lookup.layers[0]).weight
            embedding = self.lookup(
                targets.unsqueeze(-1).to(dtype=weight.dtype)
            )
        return embedding.masked_fill(~context.unsqueeze(-1), 0)
