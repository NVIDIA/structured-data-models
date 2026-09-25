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

# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D205

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, ClassVar, cast

import torch
from torch import Tensor
from torch.nn import ModuleDict

from sdm import Recipe, RelatedTables, Stype, TableTensor, Task, TaskLike
from sdm.cache import Cache
from sdm.models._huggingface import download_checkpoint
from sdm.models.base import ICLModel
from sdm.models.tabfm.ckpt import remap_ckpt
from sdm.models.tabfm.icl import ICLBlock
from sdm.models.tabfm.recipe import default_recipe
from sdm.models.tabfm.row_embedding import RowEmbedding


class TabFM(ICLModel):
    r"""The tabular foundation model from `"Introducing TabFM: A Zero-shot
    Foundation Model for Tabular Data" <https://research.google/blog/
    introducing-tabfm-a-zero-shot-foundation-model-for-tabular-data>`__.

    .. figure:: /images/tabfm_light.png
        :figclass: light-only
        :width: 100%

    .. figure:: /images/tabfm_dark.png
        :figclass: dark-only
        :width: 100%

    Architecturally, :class:`TabFM` can be viewed as a scaled-up
    :class:`TabICLv2`-style model with Fourier cell embeddings, separate
    numerical and categorical cell projections, two repeated column/row
    interaction stages, :class:`~torch.nn.RMSNorm`-based transformer blocks and
    :class:`~sdm.nn.SwiGLU` feed-forward blocks.

    .. note::
        :class:`TabFM` model weights are distributed under the
        `TabFM Non-Commercial License v1.0 <https://huggingface.co/google/
        tabfm-1.0.0-pytorch/blob/main/LICENSE>`__.
        Before downloading pretrained weights, users must accept the license
        either interactively when prompted or explicitly via
        ``accept_license=True``.

    Args:
        task: The tasks to initialize. If ``None``, all tasks supported by this
            model are initialized. Pass a single task to avoid initializing
            separate ~1.6B parameter models.
        pretrained: Whether to load the pretrained checkpoint.
        accept_license: Whether to accept the `TabFM Non-Commercial License
            v1.0 <https://huggingface.co/google/tabfm-1.0.0-pytorch/blob/main/
            LICENSE>`__ without showing the interactive license prompt.
        device: The device.
    """

    supported_feature_stypes: ClassVar[frozenset[Stype]] = frozenset(
        {Stype.numerical}
    )
    supported_target_stypes: ClassVar[frozenset[Stype]] = frozenset(
        {Stype.numerical, Stype.categorical}
    )
    supports_multi_target: ClassVar[bool] = False
    supports_related_tables: ClassVar[bool] = False

    def __init__(
        self,
        task: TaskLike | Iterable[TaskLike] | None,
        pretrained: bool = True,
        accept_license: bool = False,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__(task=task)

        self.models: ModuleDict[TaskLike, torch.nn.Module] = ModuleDict()
        for task in self.tasks:
            self.models[task] = _TabFM(
                num_classes=10 if task == Task.classification else 0,
                device="meta" if pretrained else device,
            )

        if pretrained:
            self._load_from_pretrained(accept_license, device=device)

        self.eval()

    @classmethod
    def default_recipe(cls) -> Recipe:
        r""":meta private:"""  # noqa: D415
        return default_recipe()

    def _load_from_pretrained(
        self,
        accept_license: bool,
        device: torch.device | str | None,
    ) -> TabFM:
        from safetensors.torch import load_file  # noqa: PLC0415

        TABFM_LICENSE_PROMPT = (
            "TabFM pretrained weights are distributed under the TabFM "
            "Non-Commercial License v1.0 and may be used only for "
            "non-commercial, non-production purposes. Review the license at "
            "'https://huggingface.co/google/tabfm-1.0.0-pytorch/blob/main/"
            "LICENSE' before downloading."
        )

        device = torch.get_default_device() if device is None else device

        for task, model in self.models.items():
            if task == Task.classification:
                filename = "classification/model.safetensors"
            else:
                assert task == Task.regression
                filename = "regression/model.safetensors"

            path = download_checkpoint(
                "google/tabfm-1.0.0-pytorch",
                filename,
                license_prompt=None
                if accept_license
                else TABFM_LICENSE_PROMPT,
            )
            ckpt = load_file(path, device=str(device))
            ckpt = remap_ckpt(ckpt, is_classifier=task == Task.classification)
            model.load_state_dict(ckpt, assign=True)

        return self

    def _forward(
        self,
        x_context: TableTensor | None,  # [..., R_context, D]
        y_context: TableTensor | None,  # [..., R_context, 1]
        x_query: TableTensor | None,  # [..., R_query, D]
        related_context_tables: RelatedTables[TableTensor] | None,
        related_query_tables: RelatedTables[TableTensor] | None,
        cache: Cache | None,
        generator: torch.Generator | None,
        *,
        categorical_mask: Tensor,  # [C] or [E, 1, ..., C]
        **kwargs: Any,
    ) -> TableTensor:  # [..., R_query, num_classes or 1]

        if x_query is None and x_context is not None:
            x = x_context.numerical
        elif x_context is None and x_query is not None:
            x = x_query.numerical
        else:
            assert x_context is not None
            assert x_query is not None
            x = torch.cat([x_context.numerical, x_query.numerical], dim=-2)

        classes: Tensor | None = None
        if y_context is not None and y_context.categorical.size(-1) > 0:
            y = y_context.categorical.code.squeeze(-1)
            classes = y_context.categorical.categories[0]
        elif y_context is not None and y_context.numerical.size(-1) > 0:
            y = y_context.numerical.squeeze(-1)
        else:
            assert cache is not None
            classes = cast(Tensor | None, cache["classes"])
            y = x.new_empty(
                (*x.size()[:-2], 0),
                dtype=torch.int64 if classes is not None else x.dtype,
            )

        if classes is not None and len(classes) > 10:
            raise ValueError(
                f"{self.__class__.__name__!r} only supports up to 10 classes "
                f"(got {len(classes)})"
            )

        categorical_mask = categorical_mask.expand(*x.size()[:-2], -1)

        task = Task.classification if classes is not None else Task.regression
        out = self.models[task](x, y, categorical_mask, cache=cache)

        if classes is None:
            return TableTensor(
                columns={Stype.numerical: ["pred"]},
                numerical=out,
            )

        return TableTensor(
            columns={Stype.numerical: [str(i) for i in classes.tolist()]},
            numerical=out[..., : len(classes)],
        )


class _TabFM(torch.nn.Module):
    def __init__(
        self,
        num_classes: int,
        channels: int = 256,
        num_embedding_layers: int = 3,
        num_embedding_repeats: int = 2,
        num_embedding_col_heads: int = 4,
        num_embedding_row_heads: int = 8,
        group_size: int = 3,
        num_frequencies: int = 32,
        num_inducing_points: int = 256,
        num_readout_tokens: int = 8,
        num_icl_layers: int = 24,
        num_icl_heads: int = 8,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}

        self.row_embedding = RowEmbedding(
            num_classes=num_classes,
            channels=channels,
            num_layers=num_embedding_layers,
            num_repeats=num_embedding_repeats,
            num_col_heads=num_embedding_col_heads,
            num_row_heads=num_embedding_row_heads,
            group_size=group_size,
            num_frequencies=num_frequencies,
            num_inducing_points=num_inducing_points,
            num_readout_tokens=num_readout_tokens,
            **factory_kwargs,
        )
        self.icl_block = ICLBlock(
            num_classes=num_classes,
            out_channels=max(1, num_classes),
            channels=num_readout_tokens * channels,
            num_layers=num_icl_layers,
            num_heads=num_icl_heads,
            **factory_kwargs,
        )

    def forward(
        self,
        x: Tensor,  # [..., R, C]
        y: Tensor,  # [..., R_train]
        categorical_mask: Tensor,  # [..., C]
        *,
        cache: Cache | None = None,
    ) -> Tensor:  # [..., R_test, 1 or num_classes]
        x = self.row_embedding(x, y, categorical_mask, cache=cache)
        return self.icl_block(x, y, cache=cache)
