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
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor
from torch.nn import Parameter

from sdm import Stype, TableTensor
from sdm.cache import Cache
from sdm.models.base import BaseModel
from sdm.models.tabfm.attention import MultiheadAttentionBlock
from sdm.models.tabfm.embedding import CellEmbedder, ColEmbedding
from sdm.models.tabfm.icl import ICLearning
from sdm.models.tabfm.row_interaction import RowInteraction
from sdm.processing import Recipe

_ROW_CHUNK_SIZE = 4096
_COL_CHUNK_SIZE = 16
_FFN_CHUNK_SIZE = 8192


class TabFMCore(torch.nn.Module):
    """Assemble the checkpoint-compatible TabFM neural architecture.

    The core returns predictions for every padded row, matching the upstream
    checkpoint model. A later SDM-facing wrapper is responsible for presenting
    only query/test rows through the common model API.

    Args:
        embed_dim: Number of channels in each cell token.
        max_classes: Maximum number of classification targets.
        col_num_blocks: Number of induced-attention blocks in each column
            stage.
        col_nhead: Number of column-attention heads.
        col_num_inds: Number of inducing points in each column block.
        row_num_blocks: Number of blocks in each row-interaction stage.
        row_nhead: Number of row-attention heads.
        row_num_cls: Number of learned CLS/readout tokens.
        icl_num_blocks: Number of dataset-wise ICL blocks.
        icl_nhead: Number of ICL attention heads.
        ff_factor: Multiplier used for transformer feed-forward widths.
        feature_group_size: Number of cyclically shifted features per cell
            group.
        num_freq: Number of learned Fourier frequencies per group slot.
        decoder_hidden: Hidden width of ICL target and decoder MLPs. Defaults
            to twice the flattened row representation width.
        is_classifier: Whether to emit classification logits. When ``False``,
            emit one scalar regression value per row.
        device: Device on which to create parameters and buffers.
        dtype: Dtype of parameters and buffers.
    """

    def __init__(
        self,
        *,
        embed_dim: int = 8,
        max_classes: int = 3,
        col_num_blocks: int = 2,
        col_nhead: int = 2,
        col_num_inds: int = 4,
        row_num_blocks: int = 2,
        row_nhead: int = 2,
        row_num_cls: int = 2,
        icl_num_blocks: int = 2,
        icl_nhead: int = 2,
        ff_factor: int = 2,
        feature_group_size: int = 3,
        num_freq: int = 32,
        decoder_hidden: int | None = None,
        is_classifier: bool = True,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if ff_factor <= 0:
            raise ValueError("ff_factor must be positive")

        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}
        self.max_classes = max_classes
        self.is_classifier = is_classifier
        feedforward_channels = embed_dim * ff_factor
        icl_channels = embed_dim * row_num_cls
        decoder_hidden = decoder_hidden or 2 * icl_channels

        self.cell_embedder = CellEmbedder(
            channels=embed_dim,
            max_classes=max_classes,
            feature_group_size=feature_group_size,
            num_frequencies=num_freq,
            is_classifier=is_classifier,
            **factory_kwargs,
        )
        self.col_embedder = ColEmbedding(
            channels=embed_dim,
            num_blocks=col_num_blocks,
            num_heads=col_nhead,
            feedforward_channels=feedforward_channels,
            num_inducing_points=col_num_inds,
            **factory_kwargs,
        )
        self.col_embedder_2 = ColEmbedding(
            channels=embed_dim,
            num_blocks=col_num_blocks,
            num_heads=col_nhead,
            feedforward_channels=feedforward_channels,
            num_inducing_points=col_num_inds,
            **factory_kwargs,
        )
        self.row_interactor = RowInteraction(
            channels=embed_dim,
            num_blocks=row_num_blocks,
            num_heads=row_nhead,
            feedforward_channels=feedforward_channels,
            num_cls=row_num_cls,
            output_full=True,
            **factory_kwargs,
        )
        self.row_interactor_2 = RowInteraction(
            channels=embed_dim,
            num_blocks=row_num_blocks,
            num_heads=row_nhead,
            feedforward_channels=feedforward_channels,
            num_cls=row_num_cls,
            output_full=False,
            **factory_kwargs,
        )
        self.cls_tokens = Parameter(
            torch.zeros(row_num_cls, embed_dim, **factory_kwargs)
        )
        self.icl_predictor = ICLearning(
            channels=icl_channels,
            num_blocks=icl_num_blocks,
            num_heads=icl_nhead,
            max_classes=max_classes,
            feedforward_channels=icl_channels * ff_factor,
            decoder_hidden=decoder_hidden,
            is_classifier=is_classifier,
            **factory_kwargs,
        )

        self.cell_embedder.row_chunk_size = _ROW_CHUNK_SIZE
        self.row_interactor.row_chunk_size = _ROW_CHUNK_SIZE
        self.row_interactor_2.row_chunk_size = _ROW_CHUNK_SIZE
        self.col_embedder.col_chunk_size = _COL_CHUNK_SIZE
        self.col_embedder_2.col_chunk_size = _COL_CHUNK_SIZE
        for module in self.modules():
            if isinstance(module, MultiheadAttentionBlock):
                module.ffn_chunk_size = _FFN_CHUNK_SIZE

    def forward(
        self,
        input: Tensor,
        target: Tensor,
        train_size: Tensor,
        cat_mask: Tensor | None = None,
        d: Tensor | None = None,
        cache: Cache | None = None,
    ) -> Tensor:
        """Process a batch of context and query rows through TabFM.

        Args:
            input: Input features with shape ``[B, T, H]``.
            target: Padded targets with shape ``[B, T]``. Query target values
                do not affect predictions.
            train_size: Number of context rows per table with shape ``[B]``.
            cat_mask: Optional categorical feature mask with shape ``[B, H]``.
            d: Optional active feature counts with shape ``[B]``.
            cache: Optional record/replay cache for column inducing states and
                dataset-wise context keys and values.

        Returns:
            Class logits with shape ``[B, T, K]`` for classification, or scalar
            predictions with shape ``[B, T, 1]`` for regression.
        """
        input = torch.nan_to_num(input, nan=-100.0).to(self.cls_tokens.dtype)
        embedding = self.cell_embedder(
            input,
            target,
            train_size,
            cat_mask=cat_mask,
            d=d,
        )
        embedding = self.col_embedder(
            embedding,
            train_size,
            cache=cache,
            cache_prefix="tabfm.col1",
        )

        batch_size, num_rows, _, _ = embedding.shape
        cls_tokens = self.cls_tokens.expand(
            batch_size,
            num_rows,
            -1,
            -1,
        )
        embedding = torch.cat([cls_tokens, embedding], dim=-2)
        embedding = self.row_interactor(embedding, d=d)
        embedding = self.col_embedder_2(
            embedding,
            train_size,
            cache=cache,
            cache_prefix="tabfm.col2",
        )
        representation = self.row_interactor_2(embedding, d=d)
        return self.icl_predictor(
            representation,
            target,
            train_size,
            cache=cache,
            cache_prefix="tabfm.icl",
        )


class TabFM(BaseModel):
    """Expose checkpoint-compatible TabFM cores through the SDM model API.

    Integer targets select the classification core and floating-point targets
    select the regression core. A :class:`~sdm.TableTensor` input retains its
    numerical and categorical columns; categorical indices are appended after
    numerical values and routed through an inferred categorical mask.

    When tables in a batch have different context lengths, query predictions
    are left-packed after each table's context and padded with zeros to the
    largest query count in the batch.

    Args:
        pretrained: Whether to load released TabFM weights. Loading is not yet
            performed remotely. When ``True``, ``checkpoint_path`` must point
            to an explicitly obtained local checkpoint covered by the TabFM
            Non-Commercial License v1.0.
        checkpoint_path: Local checkpoint root containing ``classification``
            and ``regression`` directories. Ignored paths are rejected, and no
            network fallback is attempted.
        cls_model: Optional classification core. Defaults to a newly
            initialized :class:`TabFMCore`.
        reg_model: Optional regression core. Defaults to a newly initialized
            :class:`TabFMCore`.
        device: Device on which to place both cores.
        dtype: Dtype to which both cores are converted.
    """

    def __init__(
        self,
        pretrained: bool = False,
        *,
        checkpoint_path: str | Path | None = None,
        cls_model: TabFMCore | None = None,
        reg_model: TabFMCore | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if pretrained:
            if checkpoint_path is None:
                raise ValueError(
                    "pretrained=True requires an explicit local "
                    "checkpoint_path; automatic TabFM downloads are disabled"
                )
            if cls_model is not None or reg_model is not None:
                raise ValueError(
                    "custom cores cannot be combined with pretrained=True"
                )
            from sdm.models.tabfm.checkpoint import _load_local_cores

            cls_model, reg_model = _load_local_cores(
                checkpoint_path,
                device=device,
                dtype=dtype,
            )
            device = None
            dtype = None
        elif checkpoint_path is not None:
            raise ValueError("checkpoint_path requires pretrained=True")

        if cls_model is None:
            cls_model = TabFMCore(
                is_classifier=True,
                device=device,
                dtype=dtype,
            )
        elif device is not None or dtype is not None:
            cls_model = cls_model.to(device=device, dtype=dtype)
        if not cls_model.is_classifier:
            raise ValueError("cls_model must be a classification TabFMCore")

        if reg_model is None:
            reg_model = TabFMCore(
                is_classifier=False,
                device=device,
                dtype=dtype,
            )
        elif device is not None or dtype is not None:
            reg_model = reg_model.to(device=device, dtype=dtype)
        if reg_model.is_classifier:
            raise ValueError("reg_model must be a regression TabFMCore")

        self.cls_model = cls_model
        self.reg_model = reg_model
        self.eval()

    @torch.inference_mode()
    def forward(
        self,
        x: Tensor | TableTensor,
        y: Tensor | TableTensor,
        *,
        train_size: Tensor | None = None,
        cat_mask: Tensor | None = None,
        d: Tensor | None = None,
    ) -> Tensor:
        """Predict query rows from context features and targets.

        Args:
            x: Features with shape ``[..., R, C]``. The rows before each
                ``train_size`` value are context rows.
            y: Context targets with shape ``[..., R_context]`` or padded
                targets with shape ``[..., R]``.
            train_size: Optional context counts with shape ``[...]``. If
                omitted, every table uses ``y.size(-1)`` context rows.
            cat_mask: Optional categorical mask with shape ``[C]`` or
                ``[..., C]``. It is inferred for :class:`~sdm.TableTensor`.
            d: Optional active feature counts with shape ``[...]`` for padded
                feature batches.

        Returns:
            Query predictions with shape ``[..., R_query, K]``. ``K`` is the
            classification core's class count for integer targets and one for
            floating-point targets. Variable query counts are zero-padded at
            the end to the largest count in the batch.
        """
        features, cat_mask = self._prepare_features(x, cat_mask=cat_mask)
        features, target = super()._preprocess(features, y)
        flat_features, batch_shape, flat_cat_mask, flat_d = (
            self._flatten_features(
                features,
                cat_mask=cat_mask,
                d=d,
            )
        )
        flat_target = target.reshape(flat_features.size(0), -1)
        flat_train_size = self._flatten_train_size(
            train_size,
            batch_shape=batch_shape,
            default=flat_target.size(-1),
            device=flat_features.device,
        )
        output = self._run_flat(
            flat_features,
            flat_target,
            flat_train_size,
            cat_mask=flat_cat_mask,
            d=flat_d,
        )
        return output.reshape(*batch_shape, output.size(-2), output.size(-1))

    @torch.inference_mode()
    def fit(
        self,
        x: Tensor | TableTensor,
        y: Tensor | TableTensor,
        *,
        cat_mask: Tensor | None = None,
        d: Tensor | None = None,
    ) -> None:
        """Prefill context attention state for subsequent prediction.

        The cache records context-derived inducing projections for both column
        stages and projected context keys and values for every ICL block.

        Args:
            x: Context features with shape ``[..., R_context, C]``.
            y: Context targets with shape ``[..., R_context]``.
            cat_mask: Optional categorical mask with shape ``[C]`` or
                ``[..., C]``. It is inferred for :class:`~sdm.TableTensor`.
            d: Optional active feature counts with shape ``[...]``.
        """
        self.clear()
        features, cat_mask = self._prepare_features(x, cat_mask=cat_mask)
        features, target = super()._preprocess(features, y)
        flat_features, batch_shape, flat_cat_mask, flat_d = (
            self._flatten_features(
                features,
                cat_mask=cat_mask,
                d=d,
            )
        )
        if target.size(-1) != features.size(-2):
            raise ValueError("fit requires one target for every context row")

        model = (
            self.reg_model if target.is_floating_point() else self.cls_model
        )
        flat_target = target.reshape(flat_features.size(0), -1)
        train_size = torch.full(
            (flat_features.size(0),),
            flat_target.size(-1),
            dtype=torch.long,
            device=flat_features.device,
        )
        cache = Cache(
            {
                "batch_shape": batch_shape,
                "cat_mask": flat_cat_mask,
                "d": flat_d,
                "y.dtype": target.dtype,
                "context_rows": flat_target.size(-1),
                "feature_count": flat_features.size(-1),
                "model.device": next(model.parameters()).device,
                "model.dtype": next(model.parameters()).dtype,
            }
        )
        model(
            flat_features,
            flat_target,
            train_size,
            cat_mask=flat_cat_mask,
            d=flat_d,
            cache=cache,
        )
        cache.freeze()
        self._cache = cache

    @torch.inference_mode()
    def predict(
        self,
        x: Tensor | TableTensor,
        *,
        cat_mask: Tensor | None = None,
        d: Tensor | None = None,
    ) -> Tensor:
        """Predict query rows using context stored by :meth:`fit`.

        Args:
            x: Query features with shape ``[..., R_query, C]``.
            cat_mask: Optional categorical mask with shape ``[C]`` or
                ``[..., C]``. It is inferred for :class:`~sdm.TableTensor`.
            d: Optional active feature counts with shape ``[...]``.

        Returns:
            Predictions with shape ``[..., R_query, K]``.
        """
        if self._cache is None:
            raise RuntimeError(
                "'TabFM' not yet fitted. Make sure to 'TabFM.fit()' "
                "beforehand."
            )

        features, cat_mask = self._prepare_features(x, cat_mask=cat_mask)
        flat_features, batch_shape, flat_cat_mask, flat_d = (
            self._flatten_features(
                features,
                cat_mask=cat_mask,
                d=d,
            )
        )
        cached_batch_shape = cast(tuple[int, ...], self._cache["batch_shape"])
        if batch_shape != cached_batch_shape:
            raise ValueError(
                "query and fitted context batch shapes must match"
            )

        cached_cat_mask = cast(Tensor | None, self._cache["cat_mask"])
        cached_d = cast(Tensor | None, self._cache["d"])
        feature_count = cast(int, self._cache["feature_count"])
        if feature_count != flat_features.size(-1):
            raise ValueError(
                "query and fitted context feature counts must match"
            )
        self._check_replay_metadata(
            "cat_mask",
            cached_cat_mask,
            flat_cat_mask,
        )
        self._check_replay_metadata("d", cached_d, flat_d)

        target_dtype = cast(torch.dtype, self._cache["y.dtype"])
        model = (
            self.reg_model
            if target_dtype.is_floating_point
            else self.cls_model
        )
        model_device = next(model.parameters()).device
        model_dtype = next(model.parameters()).dtype
        if model_device != self._cache["model.device"]:
            raise ValueError("model device changed after TabFM.fit()")
        if model_dtype != self._cache["model.dtype"]:
            raise ValueError("model dtype changed after TabFM.fit()")
        if flat_features.device != model_device:
            raise ValueError(
                "query input and fitted model must use one device"
            )

        query_target = torch.full(
            (flat_features.size(0), flat_features.size(1)),
            -100,
            dtype=target_dtype,
            device=flat_features.device,
        )
        train_size = torch.zeros(
            flat_features.size(0),
            dtype=torch.long,
            device=flat_features.device,
        )
        output = model(
            flat_features,
            query_target,
            train_size,
            cat_mask=cached_cat_mask,
            d=cached_d,
            cache=self._cache,
        )
        return output.reshape(*batch_shape, output.size(-2), output.size(-1))

    def default_recipe(self) -> Recipe:
        """Return the default single-estimator regression recipe.

        Returns:
            The context-fitted :class:`~sdm.processing.Recipe` for numerical
            features and regression targets.
        """
        from sdm.models.tabfm.recipe import default_regression_recipe

        return default_regression_recipe()

    def _forward(
        self,
        x: Tensor,
        y: Tensor,
        *,
        cache: Cache | None = None,
    ) -> Tensor:
        if cache is not None:
            raise RuntimeError(
                "TabFM caching is handled by fit() and predict()"
            )
        return self.forward(x, y)

    def _prepare_features(
        self,
        x: Tensor | TableTensor,
        *,
        cat_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor | None]:
        if not isinstance(x, TableTensor):
            return x, cat_mask
        if cat_mask is not None:
            raise ValueError("cat_mask is inferred for TableTensor input")

        unsupported = [
            stype.value
            for stype in (Stype.datetime, Stype.id)
            if x.blocks[stype].size(-1) > 0
        ]
        if unsupported:
            raise ValueError(
                "TabFM TableTensor input only supports numerical and "
                f"categorical columns, but found {'/'.join(unsupported)}"
            )

        numerical = x.numerical
        categorical = x.categorical.as_tensor()
        model_dtype = next(self.parameters()).dtype
        dtype = numerical.dtype if numerical.size(-1) > 0 else model_dtype
        features = torch.cat(
            [numerical.to(dtype), categorical.to(dtype)],
            dim=-1,
        )
        inferred_mask = torch.cat(
            [
                torch.zeros(
                    numerical.size(-1),
                    dtype=torch.bool,
                    device=x.device,
                ),
                torch.ones(
                    categorical.size(-1),
                    dtype=torch.bool,
                    device=x.device,
                ),
            ]
        )
        return features, inferred_mask

    def _flatten_features(
        self,
        features: Tensor,
        *,
        cat_mask: Tensor | None,
        d: Tensor | None,
    ) -> tuple[Tensor, tuple[int, ...], Tensor | None, Tensor | None]:
        if features.dim() < 2:
            raise ValueError("x must have shape [..., R, C]")
        if not features.is_floating_point():
            raise ValueError("x must have a floating-point dtype")
        batch_shape = tuple(features.shape[:-2])
        batch_size = math.prod(batch_shape) if batch_shape else 1
        num_features = features.size(-1)
        if num_features == 0:
            raise ValueError("x must contain at least one feature")
        flat_features = features.reshape(
            batch_size,
            features.size(-2),
            num_features,
        )

        flat_cat_mask = None
        if cat_mask is not None:
            if cat_mask.dtype != torch.bool:
                raise ValueError("cat_mask must have boolean dtype")
            if cat_mask.shape == (num_features,):
                flat_cat_mask = cat_mask.expand(batch_size, -1)
            elif cat_mask.shape == (*batch_shape, num_features):
                flat_cat_mask = cat_mask.reshape(batch_size, num_features)
            else:
                raise ValueError("cat_mask must have shape [C] or [..., C]")
            if flat_cat_mask.device != features.device:
                raise ValueError("cat_mask and x must use the same device")

        flat_d = None
        if d is not None:
            if d.is_floating_point():
                raise ValueError("d must have an integer dtype")
            if d.dim() == 0:
                flat_d = d.expand(batch_size)
            elif d.shape == batch_shape:
                flat_d = d.reshape(batch_size)
            else:
                raise ValueError("d must have shape [...]")
            if flat_d.device != features.device:
                raise ValueError("d and x must use the same device")
            if torch.any((flat_d <= 0) | (flat_d > num_features)):
                raise ValueError("d values must be in [1, C]")

        return flat_features, batch_shape, flat_cat_mask, flat_d

    def _flatten_train_size(
        self,
        train_size: Tensor | None,
        *,
        batch_shape: tuple[int, ...],
        default: int,
        device: torch.device,
    ) -> Tensor:
        batch_size = math.prod(batch_shape) if batch_shape else 1
        if train_size is None:
            return torch.full(
                (batch_size,),
                default,
                dtype=torch.long,
                device=device,
            )
        if train_size.is_floating_point():
            raise ValueError("train_size must have an integer dtype")
        if train_size.dim() == 0:
            train_size = train_size.expand(batch_size)
        elif train_size.shape == batch_shape:
            train_size = train_size.reshape(batch_size)
        else:
            raise ValueError("train_size must have shape [...]")
        if train_size.device != device:
            raise ValueError("train_size and x must use the same device")
        return train_size.to(torch.long)

    def _run_flat(
        self,
        input: Tensor,
        target: Tensor,
        train_size: Tensor,
        *,
        cat_mask: Tensor | None,
        d: Tensor | None,
    ) -> Tensor:
        num_rows = input.size(-2)
        if target.size(-1) > num_rows:
            raise ValueError("y cannot contain more rows than x")
        if torch.any(train_size < 1):
            raise ValueError("train_size values must be positive")
        if torch.any(train_size > target.size(-1)):
            raise ValueError(
                "train_size values cannot exceed available y rows"
            )
        if torch.any(train_size >= num_rows):
            raise ValueError("every table must contain at least one query row")

        if target.size(-1) < num_rows:
            padding = target.new_full(
                (target.size(0), num_rows - target.size(-1)),
                -100,
            )
            target = torch.cat([target, padding], dim=-1)

        model = (
            self.reg_model if target.is_floating_point() else self.cls_model
        )
        output = model(
            input,
            target,
            train_size,
            cat_mask=cat_mask,
            d=d,
        )

        query_offset = torch.arange(num_rows, device=input.device)
        query_offset = query_offset[query_offset < num_rows - train_size.min()]
        query_index = train_size[:, None] + query_offset[None, :]
        valid = query_index < num_rows
        query_index = query_index.clamp_max(num_rows - 1)
        query_index = query_index[..., None].expand(-1, -1, output.size(-1))
        output = output.gather(dim=1, index=query_index)
        return output.masked_fill(~valid[..., None], 0)

    @staticmethod
    def _check_replay_metadata(
        name: str,
        cached: Tensor | None,
        query: Tensor | None,
    ) -> None:
        if cached is None and query is None:
            return
        if cached is None or query is None or not torch.equal(cached, query):
            raise ValueError(
                f"query {name} must match the fitted context {name}"
            )

    def __repr__(self) -> str:
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        args = []
        if device.type != "cpu":
            args.append(f"device={device}")
        if dtype != torch.float32:
            args.append(f"dtype={dtype}")
        return f"{self.__class__.__name__}({', '.join(args)})"
