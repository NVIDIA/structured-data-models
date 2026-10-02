# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D205
from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any, ClassVar, Literal, cast

import torch
from torch import Tensor
from torch.nn import Identity, Linear, ModuleDict

from sdm import Recipe, RelatedTables, Stype, TableTensor, Task, TaskLike
from sdm.cache import Cache
from sdm.models import ECOC, ICLModel
from sdm.models._huggingface import download_checkpoint
from sdm.models.kumo.tabular.block import KumoTabularTransformerBlock
from sdm.models.kumo.tabular.icl import ICLBlock
from sdm.models.kumo.tabular.recipe import default_recipe
from sdm.models.kumo.tabular.row_embedding import RowEmbedding

MODEL_KWARGS: dict[str, dict[str, Any]] = {
    "small": {
        "cell_channels": 128,
        "num_embedding_layers": 4,
        "num_embedding_heads": 4,
        "num_inducing_points": 128,
        "group_size": 3,
        "num_frequencies": 32,
        "num_readout_tokens": 4,
        "icl_channels": 512,
        "num_icl_layers": 12,
        "num_icl_heads": 8,
        "num_icl_key_value_heads_for_query": None,
    },
    "medium": {
        "cell_channels": 256,
        "num_embedding_layers": 6,
        "num_embedding_heads": 4,
        "num_inducing_points": 256,
        "group_size": 3,
        "num_frequencies": 32,
        "num_readout_tokens": 4,
        "icl_channels": 512,
        "num_icl_layers": 24,
        "num_icl_heads": 8,
        "num_icl_key_value_heads_for_query": 2,
    },
    "large": {
        "cell_channels": 256,
        "num_embedding_layers": 6,
        "num_embedding_heads": 4,
        "num_inducing_points": 256,
        "group_size": 3,
        "num_frequencies": 32,
        "num_readout_tokens": 4,
        "icl_channels": 1024,
        "num_icl_layers": 24,
        "num_icl_heads": 16,
        "num_icl_key_value_heads_for_query": 2,
    },
}


class KumoTabular(ICLModel):
    r"""The tabular foundation model from `"NVIDIA Kumo Tabular Sets a New
    Accuracy-Efficiency Frontier for Tabular Prediction"
    <https://huggingface.co/blog/nvidia/kumo-tabular>`__.

    .. figure:: /images/kumo_tabular.svg
        :width: 100%

    :class:`KumoTabular` processes a table in two stages. First, an interleaved
    row/column encoder transforms raw table cells into fixed-size row
    representations. Numerical and categorical cells use separate learned
    Fourier frequencies and projections, together with a learned missingness
    projection. Each encoder stage consists of:

    * **Column-wise:** Each feature group is processed across rows using
      induced set attention. Both context and query cells attend only to
      context-row keys and values.
    * **Row-wise:** Each row's feature groups and learnable readout tokens
      attend to one another, combining feature interactions into a fixed-size
      row representation.

    Second, a dataset-wise in-context learning transformer processes the row
    representations and predicts each query target from the labeled context
    rows. The ``"large"`` model widens this transformer to 1024 channels and
    16 attention heads. Its four 256-channel readout tokens concatenate
    directly to that width without a projection layer.

    For regression tasks, :class:`KumoTabular` predicts 999 quantiles named
    ``"q001"`` through ``"q999"``.

    Args:
        task: The tasks to initialize. If ``None``, all tasks supported by this
            model are initialized.
        size: The model size, one of ``"small"``, ``"medium"``, or
            ``"large"``. Defaults to ``"large"``.
        pretrained: Whether to load pretrained checkpoints.
        device: The device for model parameters. If ``None``, uses PyTorch's
            default device.
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
        task: TaskLike | Iterable[TaskLike] | None = None,
        size: Literal["small", "medium", "large"] = "large",
        pretrained: bool = True,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__(task=task)
        if Task.classification in self.tasks:
            self.ecoc = ECOC(max_classes=10)

        self.models: ModuleDict[TaskLike, torch.nn.Module] = ModuleDict()
        for task in self.tasks:
            self.models[task] = _KumoTabular(
                num_classes=10 if task == Task.classification else 0,
                num_quantiles=999 if task == Task.regression else 0,
                device="meta" if pretrained else device,
                **MODEL_KWARGS[size],
            )

        if pretrained:
            self.models[task] = self._load_from_pretrained(size, device=device)

        self.eval()

    def estimate_estimator_batch_size(
        self,
        *,
        num_rows: int,
        num_columns: int,
        num_estimators: int,
        memory_budget: int,
        num_classes: int = 0,
        dtype: torch.dtype = torch.float16,
    ) -> int:
        """Estimate the estimator batch size for fitting the default recipe.

        This inference heuristic reserves half the budget for unmodeled
        allocations and returns at least one estimator, even when it may not
        fit. It is not an OOM guarantee. Custom recipes and autograd execution
        require their own estimates. No model state or batching is changed.

        Args:
            num_rows: Number of context rows.
            num_columns: Number of input feature columns before preprocessing.
            num_estimators: Total number of ensemble members.
            memory_budget: Available device bytes after weights and inputs.
            num_classes: Target class count, including absent classes; zero
                selects regression.
            dtype: Neural execution dtype, including any autocast setting.

        Returns:
            Estimator batch size between one and ``num_estimators``.
        """
        workspace, cache = self._estimate_row_bytes(
            num_columns=num_columns, num_classes=num_classes, dtype=dtype
        )
        capacity = (memory_budget // 2) // max(
            num_rows * (workspace + cache), 1
        )
        return max(1, min(num_estimators, capacity))

    def estimate_query_batch_size(
        self,
        *,
        num_columns: int,
        num_estimators: int,
        estimator_batch_size: int,
        memory_budget: int,
        num_classes: int = 0,
        dtype: torch.dtype = torch.float16,
    ) -> int:
        """Estimate how many query rows to predict with the default recipe.

        This inference heuristic reserves half the budget for unmodeled
        allocations and returns at least one row, even when it may not fit.
        It is not an OOM guarantee. Custom recipes and autograd execution
        require their own estimates. No model state or batching is changed.

        Args:
            num_columns: Number of input feature columns before preprocessing.
            num_estimators: Total number of ensemble members.
            estimator_batch_size: Maximum number of estimators run together.
            memory_budget: Available device bytes after weights, inputs, and
                resident or staged context caches, including transfer overlap.
            num_classes: Target class count, including absent classes; zero
                selects regression with 999 output quantiles.
            dtype: Neural execution dtype, including any autocast setting.

        Returns:
            Query row batch size of at least one. Cap it to the query size
            and any runner-specific memory limit before applying it.
        """
        workspace, _ = self._estimate_row_bytes(
            num_columns=num_columns, num_classes=num_classes, dtype=dtype
        )
        output_columns = num_classes or 999
        # Preprocessing and output reduction retain all ensemble members.
        row_bytes = workspace * estimator_batch_size
        row_bytes += (
            num_estimators * 8 * (8 * num_columns + 4 * output_columns)
        )
        return max(1, (memory_budget // 2) // max(row_bytes, 1))

    def _estimate_row_bytes(
        self,
        *,
        num_columns: int,
        num_classes: int,
        dtype: torch.dtype,
    ) -> tuple[int, int]:
        task = Task.classification if num_classes else Task.regression
        model = cast(_KumoTabular, self.models[task])
        row = model.row_embedding
        icl = model.icl_block
        layer = cast(KumoTabularTransformerBlock, icl.layers[0])
        tasks = 1
        if num_classes and num_classes > self.ecoc.max_classes:
            tasks = max(
                math.ceil(num_classes / (self.ecoc.max_classes - 1)),
                4 * math.ceil(math.log(num_classes, self.ecoc.max_classes)),
            )
        # The default recipe adds count columns and keeps at most 500 features.
        columns = min(2 * num_columns, 500)
        element_size = dtype.itemsize
        workspace = tasks * (
            element_size
            * 4
            * (columns + row.readout_token.size(0))
            * row.channels
            + layer.peak_bytes_per_example(
                element_size=element_size, query_length=1
            )
        )
        # Fit projects all heads before retaining the smaller query KV heads.
        cache = tasks * element_size * 2 * layer.attn.q_dim * len(icl.layers)
        return workspace, cache

    @classmethod
    def default_recipe(cls) -> Recipe:
        r""":meta private:"""  # noqa: D415
        return default_recipe()

    def _load_from_pretrained(
        self,
        size: Literal["small", "medium", "large"],
        device: torch.device | str | None,
    ) -> _KumoTabular:
        device = torch.get_default_device() if device is None else device

        for task, model in self.models.items():
            if task == Task.classification:
                filename = f"{size}/classifier.pt"
            else:
                assert task == Task.regression
                filename = f"{size}/regressor.pt"

            path = download_checkpoint(
                repo_id="nvidia/Kumo-Tabular",
                filename=filename,
                revision="v1.0.0",
            )
            ckpt = torch.load(path, map_location=device, weights_only=True)
            model.load_state_dict(ckpt, assign=True)

        return model

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
        categorical_mask: Tensor,
        **kwargs: Any,
    ) -> TableTensor:  # [..., R_query, num_classes or 999]

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

        categorical_mask = categorical_mask.expand(*x.size()[:-2], -1)

        if classes is None:
            out = self.models[Task.regression](
                x=x,
                y=y,
                categorical_mask=categorical_mask,
                cache=cache,
            )
            return TableTensor(
                columns={
                    Stype.numerical: [f"q{i:03d}" for i in range(1, 1000)]
                },
                numerical=out,
            )

        out = self.ecoc(
            model=self.models[Task.classification],
            x=x,
            y=y,
            num_classes=len(classes),
            num_members=x.size(0) if x.dim() > 2 else 1,
            cache=cache,
            generator=generator,
            categorical_mask=categorical_mask,
        )
        return TableTensor(
            columns={Stype.numerical: [str(i) for i in classes.tolist()]},
            numerical=out,
        )


class _KumoTabular(torch.nn.Module):
    def __init__(
        self,
        num_classes: int,
        num_quantiles: int,
        cell_channels: int = 128,
        num_embedding_layers: int = 4,
        num_embedding_heads: int = 4,
        num_inducing_points: int = 128,
        group_size: int = 3,
        num_frequencies: int = 32,
        num_readout_tokens: int = 4,
        icl_channels: int = 512,
        num_icl_layers: int = 12,
        num_icl_heads: int = 8,
        num_icl_key_value_heads_for_query: int | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}

        self.row_embedding = RowEmbedding(
            num_classes=num_classes,
            channels=cell_channels,
            num_layers=num_embedding_layers,
            num_heads=num_embedding_heads,
            group_size=group_size,
            num_frequencies=num_frequencies,
            num_inducing_points=num_inducing_points,
            num_readout_tokens=num_readout_tokens,
            **factory_kwargs,
        )
        if cell_channels * num_readout_tokens != icl_channels:
            self.row_project = Linear(
                cell_channels * num_readout_tokens,
                icl_channels,
                **factory_kwargs,
            )
        else:
            self.row_project = Identity()
        self.icl_block = ICLBlock(
            num_classes=num_classes,
            out_channels=num_classes or num_quantiles,
            channels=icl_channels,
            num_layers=num_icl_layers,
            num_heads=num_icl_heads,
            num_key_value_heads_for_query=num_icl_key_value_heads_for_query,
            **factory_kwargs,
        )

    def forward(
        self,
        x: Tensor,  # [..., R, C]
        y: Tensor,  # [..., R_train]
        categorical_mask: Tensor,  # [..., C]
        *,
        cache: Cache | None = None,
    ) -> Tensor:  # [..., R_test, num_classes or num_quantiles]
        x = self.row_embedding(x, y, categorical_mask, cache=cache)
        x = self.row_project(x)
        return self.icl_block(x=x, y=y, cache=cache)
