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
from torch.nn import Embedding, Linear, ModuleList, Parameter

from sdm.cache import Cache, KVCacheEntry
from sdm.models.tabfm.attention import MultiheadAttentionBlock, RMSNorm
from sdm.models.tabfm.mlp import MLP


class InducedSelfAttentionBlock(torch.nn.Module):
    r"""Induced attention from the `Set Transformer`_ paper.

    Learned inducing vectors first attend to the input set, after which the
    input elements attend to the resulting fixed-size representation:

    .. math::

        H = \operatorname{MAB}(I, X), \qquad
        Y = \operatorname{MAB}(X, H).

    .. _Set Transformer: https://arxiv.org/abs/1810.00825

    Args:
        channels: Number of input and output channels.
        num_heads: Number of attention heads.
        feedforward_channels: Hidden width of each feed-forward network.
        num_inducing_points: Number of learned inducing vectors.
        activation: Feed-forward activation used by both attention blocks.
        device: Device on which to create parameters.
        dtype: Dtype of the parameters.
    """

    def __init__(
        self,
        channels: int,
        num_heads: int,
        feedforward_channels: int,
        num_inducing_points: int,
        activation: str = "swiglu",
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if num_inducing_points <= 0:
            raise ValueError("num_inducing_points must be positive")

        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}
        self.ind_vectors = Parameter(
            torch.zeros(
                num_inducing_points,
                channels,
                **factory_kwargs,
            )
        )
        self.mab1 = MultiheadAttentionBlock(
            channels=channels,
            num_heads=num_heads,
            feedforward_channels=feedforward_channels,
            activation=activation,
            **factory_kwargs,
        )
        self.mab2 = MultiheadAttentionBlock(
            channels=channels,
            num_heads=num_heads,
            feedforward_channels=feedforward_channels,
            activation=activation,
            **factory_kwargs,
        )

    def forward(
        self,
        input: Tensor,
        attn_mask: Tensor | None = None,
        key_value: KVCacheEntry | None = None,
        return_key_value: bool = False,
    ) -> Tensor | tuple[Tensor, KVCacheEntry]:
        """Process an input set through learned inducing vectors.

        Args:
            input: Input tensor with shape ``[..., S, D]``. ``S`` is the set
                size and ``D`` is ``channels``.
            attn_mask: Boolean or additive mask broadcastable to
                ``[..., H, M, S]``, where ``M`` is ``num_inducing_points``.
            key_value: Optional cached projections of context-derived inducing
                states.
            return_key_value: Whether to return inducing-state projections for
                later replay.

        Returns:
            Tensor with shape ``[..., S, D]`` and, when requested, cached
            inducing-state projections.
        """
        *batch_shape, _, channels = input.shape
        if channels != self.ind_vectors.size(-1):
            raise ValueError("input channels do not match inducing vectors")
        if key_value is None:
            inducing_points = self.ind_vectors.view(
                *(1,) * len(batch_shape),
                *self.ind_vectors.shape,
            ).expand(
                *batch_shape,
                *self.ind_vectors.shape,
            )
            hidden = cast(
                Tensor,
                self.mab1(
                    inducing_points,
                    input,
                    input,
                    attn_mask=attn_mask,
                ),
            )
            return self.mab2(
                input,
                hidden,
                hidden,
                return_key_value=return_key_value,
            )
        return self.mab2(input, key=key_value)


class SetTransformer(torch.nn.Module):
    """Stack induced attention blocks from the Set Transformer architecture.

    Args:
        num_blocks: Number of induced self-attention blocks.
        channels: Number of input and output channels.
        num_heads: Number of attention heads.
        feedforward_channels: Hidden width of each feed-forward network.
        num_inducing_points: Number of learned inducing vectors per block.
        activation: Feed-forward activation used by every block.
        device: Device on which to create parameters.
        dtype: Dtype of the parameters.
    """

    def __init__(
        self,
        num_blocks: int,
        channels: int,
        num_heads: int,
        feedforward_channels: int,
        num_inducing_points: int,
        activation: str = "swiglu",
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if num_blocks <= 0:
            raise ValueError("num_blocks must be positive")

        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}
        self.blocks = ModuleList(
            InducedSelfAttentionBlock(
                channels=channels,
                num_heads=num_heads,
                feedforward_channels=feedforward_channels,
                num_inducing_points=num_inducing_points,
                activation=activation,
                **factory_kwargs,
            )
            for _ in range(num_blocks)
        )

    def forward(
        self,
        input: Tensor,
        attn_mask: Tensor | None = None,
        cache: Cache | None = None,
        cache_prefix: str = "set_transformer",
    ) -> Tensor:
        """Encode an input set through the induced attention stack.

        Args:
            input: Input tensor with shape ``[..., S, D]``.
            attn_mask: Boolean or additive mask broadcastable to
                ``[..., H, M, S]``.
            cache: Optional record/replay cache for inducing-state
                projections.
            cache_prefix: Key namespace used within ``cache``.

        Returns:
            Tensor with shape ``[..., S, D]``.
        """
        for index, block in enumerate(self.blocks):
            key = f"{cache_prefix}.block{index}"
            key_value = (
                cast(KVCacheEntry, cache[key])
                if cache is not None and cache.is_replaying
                else None
            )
            result = block(
                input,
                attn_mask=attn_mask,
                key_value=key_value,
                return_key_value=cache is not None and cache.is_recording,
            )
            if cache is not None and cache.is_recording:
                input, cache[key] = cast(
                    tuple[Tensor, KVCacheEntry],
                    result,
                )
            else:
                input = cast(Tensor, result)
        return input


class CellEmbedder(torch.nn.Module):
    """Embed grouped numerical and categorical cells for TabFM.

    Each feature position gathers a cyclic group with offsets
    ``2**index - 1``. Numerical and categorical group slots use separate
    learned Fourier frequencies and linear projections. Target embeddings are
    added only to context rows, and padded columns are zeroed when ``d`` is
    supplied.

    Args:
        channels: Number of output channels per embedded cell.
        max_classes: Maximum number of classification targets.
        feature_group_size: Number of cyclically shifted features per group.
        num_frequencies: Number of learned Fourier frequencies per group slot.
        is_classifier: Whether to use a class lookup instead of a regression
            target MLP.
        device: Device on which to create parameters and buffers.
        dtype: Dtype of parameters and buffers.
    """

    fourier_frequencies: Tensor
    fourier_frequencies_cat: Tensor

    def __init__(
        self,
        channels: int,
        max_classes: int,
        feature_group_size: int = 3,
        num_frequencies: int = 32,
        is_classifier: bool = True,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        if max_classes <= 0:
            raise ValueError("max_classes must be positive")
        if feature_group_size <= 0:
            raise ValueError("feature_group_size must be positive")
        if num_frequencies <= 0:
            raise ValueError("num_frequencies must be positive")

        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}
        self.embed_dim = channels
        self.fgs = feature_group_size
        self.is_classifier = is_classifier
        self.register_buffer(
            "fourier_frequencies",
            torch.zeros(
                feature_group_size,
                num_frequencies,
                **factory_kwargs,
            ),
        )
        self.register_buffer(
            "fourier_frequencies_cat",
            torch.zeros(
                feature_group_size,
                num_frequencies,
                **factory_kwargs,
            ),
        )
        self.in_linear = Linear(
            2 * num_frequencies,
            channels,
            **factory_kwargs,
        )
        self.in_linear_cat = Linear(
            2 * num_frequencies,
            channels,
            **factory_kwargs,
        )
        self.y_embedder_lookup: Embedding | MLP
        if is_classifier:
            self.y_embedder_lookup = Embedding(
                max_classes,
                channels,
                **factory_kwargs,
            )
        else:
            self.y_embedder_lookup = MLP(
                in_channels=1,
                hidden_channels=[6],
                out_channels=channels,
                activation="gelu",
                **factory_kwargs,
            )
        self.row_chunk_size: int | None = None

    def _group(self, input: Tensor, d: Tensor | None = None) -> Tensor:
        batch_size, num_rows, num_features = input.shape
        feature_index = torch.arange(num_features, device=input.device)
        offsets = 2 ** torch.arange(self.fgs, device=input.device) - 1
        if d is None:
            index = (
                feature_index.unsqueeze(-1) + offsets.unsqueeze(0)
            ) % num_features
            return input[..., index]

        safe_features = d.to(torch.long).clamp_min(1)
        index = (
            feature_index[None, :, None] + offsets[None, None, :]
        ) % safe_features[:, None, None]
        index = index[:, None, :, :].expand(
            batch_size,
            num_rows,
            num_features,
            self.fgs,
        )
        expanded = input.unsqueeze(-1).expand(
            batch_size,
            num_rows,
            num_features,
            self.fgs,
        )
        return expanded.gather(dim=-2, index=index)

    def _cell(
        self,
        input: Tensor,
        cat_mask: Tensor | None,
        d: Tensor | None = None,
    ) -> Tensor:
        grouped = self._group(input, d=d).unsqueeze(-1).float()
        input_dtype = input.dtype

        numerical_angles = grouped * self.fourier_frequencies.float()
        numerical_fourier = torch.cat(
            [numerical_angles.sin(), numerical_angles.cos()],
            dim=-1,
        ).to(input_dtype)
        numerical = self.in_linear(numerical_fourier)
        if cat_mask is None:
            return numerical.sum(dim=-2)

        categorical_angles = grouped * self.fourier_frequencies_cat.float()
        categorical_fourier = torch.cat(
            [categorical_angles.sin(), categorical_angles.cos()],
            dim=-1,
        ).to(input_dtype)
        categorical = self.in_linear_cat(categorical_fourier)
        grouped_cat_mask = self._group(
            cat_mask[:, None, :].to(torch.float32),
            d=d,
        ).bool()[..., None]
        return torch.where(
            grouped_cat_mask,
            categorical,
            numerical,
        ).sum(dim=-2)

    def forward(
        self,
        input: Tensor,
        target: Tensor,
        train_size: Tensor,
        cat_mask: Tensor | None = None,
        d: Tensor | None = None,
    ) -> Tensor:
        """Embed feature cells and inject targets into context rows.

        Args:
            input: Input features with shape ``[B, T, H]``.
            target: Padded targets with shape ``[B, T]``. Values at and after
                each table's ``train_size`` do not affect the output.
            train_size: Number of context rows per table with shape ``[B]``.
            cat_mask: Optional categorical feature mask with shape ``[B, H]``.
            d: Optional active feature counts with shape ``[B]``. Group indices
                wrap by these counts, and output columns at or after each count
                are zeroed.

        Returns:
            Cell embeddings with shape ``[B, T, H, D]``.
        """
        if input.dim() != 3:
            raise ValueError("input must have shape [B, T, H]")
        if not input.is_floating_point():
            raise ValueError("input must have a floating-point dtype")
        batch_size, num_rows, num_features = input.shape
        if num_rows == 0 or num_features == 0:
            raise ValueError("input must contain at least one row and feature")
        if target.shape != (batch_size, num_rows):
            raise ValueError("target must have shape [B, T]")
        if target.device != input.device:
            raise ValueError("target and input must use the same device")
        if train_size.shape != (batch_size,):
            raise ValueError("train_size must have shape [B]")
        if train_size.is_floating_point():
            raise ValueError("train_size must have an integer dtype")
        if train_size.device != input.device:
            raise ValueError("train_size and input must use the same device")
        if cat_mask is not None:
            if cat_mask.shape != (batch_size, num_features):
                raise ValueError("cat_mask must have shape [B, H]")
            if cat_mask.dtype != torch.bool:
                raise ValueError("cat_mask must have a boolean dtype")
            if cat_mask.device != input.device:
                raise ValueError("cat_mask and input must use the same device")
        if d is not None:
            if d.shape != (batch_size,):
                raise ValueError("d must have shape [B]")
            if d.is_floating_point():
                raise ValueError("d must have an integer dtype")
            if d.device != input.device:
                raise ValueError("d and input must use the same device")

        if self.row_chunk_size is None:
            cell = self._cell(input, cat_mask, d=d)
        else:
            if self.row_chunk_size <= 0:
                raise ValueError("row_chunk_size must be positive or None")
            # Rows are independent during cell embedding, so this loop bounds
            # the Fourier activation without changing row context.
            cell = torch.cat(
                [
                    self._cell(
                        input[:, start : start + self.row_chunk_size],
                        cat_mask,
                        d=d,
                    )
                    for start in range(0, num_rows, self.row_chunk_size)
                ],
                dim=1,
            )

        if self.is_classifier:
            target_embedder = cast(Embedding, self.y_embedder_lookup)
            clean_target = target.long().clamp(
                0,
                target_embedder.num_embeddings - 1,
            )
            target_embedding = target_embedder(clean_target)
        else:
            target_embedder = cast(MLP, self.y_embedder_lookup)
            target_embedding = target_embedder(
                target[..., None].to(cell.dtype)
            )

        row_index = torch.arange(num_rows, device=input.device)
        train_mask = row_index[None, :] < train_size[:, None]
        train_mask = train_mask[..., None, None]  # [B, T, 1, 1].
        output = torch.where(
            train_mask,
            cell + target_embedding[:, :, None, :],
            cell,
        )

        if d is not None:
            feature_index = torch.arange(num_features, device=input.device)
            feature_mask = feature_index[None, :] < d[:, None]
            feature_mask = feature_mask[:, None, :, None]  # [B, 1, H, 1].
            output = torch.where(
                feature_mask,
                output,
                torch.zeros_like(output),
            )
        return output


class ColEmbedding(torch.nn.Module):
    """Build distribution-aware embeddings independently for every column.

    Columns are folded into the batch axis and rows become the set dimension.
    Inducing points attend only to context rows selected by ``train_size``;
    query rows then attend to that context-derived inducing representation.

    Args:
        channels: Number of input and output channels.
        num_blocks: Number of induced self-attention blocks.
        num_heads: Number of attention heads.
        feedforward_channels: Hidden width of each feed-forward network.
        num_inducing_points: Number of learned inducing vectors per block.
        activation: Feed-forward activation used by every block.
        device: Device on which to create parameters.
        dtype: Dtype of the parameters.
    """

    def __init__(
        self,
        channels: int,
        num_blocks: int,
        num_heads: int,
        feedforward_channels: int,
        num_inducing_points: int,
        activation: str = "swiglu",
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}
        self.tf_col = SetTransformer(
            num_blocks=num_blocks,
            channels=channels,
            num_heads=num_heads,
            feedforward_channels=feedforward_channels,
            num_inducing_points=num_inducing_points,
            activation=activation,
            **factory_kwargs,
        )
        self.out_w = Linear(channels, channels, **factory_kwargs)
        self.ln_w = RMSNorm(channels, **factory_kwargs)
        self.col_chunk_size: int | None = None

    def _stage(
        self,
        input: Tensor,
        attn_mask: Tensor | None,
        cache: Cache | None = None,
        cache_prefix: str = "col_embedding",
    ) -> Tensor:
        input = self.tf_col(
            input,
            attn_mask=attn_mask,
            cache=cache,
            cache_prefix=cache_prefix,
        )
        return self.ln_w(self.out_w(input))

    def _cached_chunked_stage(
        self,
        input: Tensor,
        attn_mask: Tensor | None,
        *,
        cache: Cache,
        cache_prefix: str,
        chunk_size: int,
    ) -> Tensor:
        cache_keys = [
            f"{cache_prefix}.block{index}"
            for index in range(len(self.tf_col.blocks))
        ]
        output = []
        recorded: dict[str, list[KVCacheEntry]] = {
            key: [] for key in cache_keys
        }
        for start in range(0, input.size(0), chunk_size):
            stop = start + chunk_size
            if cache.is_recording:
                chunk_cache = Cache()
            else:
                chunk_entries = {}
                for key in cache_keys:
                    entry = cast(KVCacheEntry, cache[key])
                    chunk_entries[key] = KVCacheEntry(
                        key=entry.key[start:stop],
                        value=entry.value[start:stop],
                    )
                chunk_cache = Cache(chunk_entries)
                chunk_cache.freeze()

            output.append(
                self._stage(
                    input[start:stop],
                    None if attn_mask is None else attn_mask[start:stop],
                    cache=chunk_cache,
                    cache_prefix=cache_prefix,
                )
            )
            if cache.is_recording:
                for key in cache_keys:
                    recorded[key].append(cast(KVCacheEntry, chunk_cache[key]))

        if cache.is_recording:
            for key, entries in recorded.items():
                cache[key] = KVCacheEntry(
                    key=torch.cat([entry.key for entry in entries], dim=0),
                    value=torch.cat(
                        [entry.value for entry in entries],
                        dim=0,
                    ),
                )
        return torch.cat(output, dim=0)

    def forward(
        self,
        input: Tensor,
        train_size: Tensor,
        cache: Cache | None = None,
        cache_prefix: str = "col_embedding",
    ) -> Tensor:
        """Embed columns using statistics from context rows.

        Args:
            input: Cell embeddings with shape ``[B, T, H, D]``. ``B`` is the
                number of tables, ``T`` is the padded row count, ``H`` is the
                number of feature groups, and ``D`` is ``channels``.
            train_size: Number of context rows per table with shape ``[B]``.
            cache: Optional record/replay cache for column inducing states.
            cache_prefix: Key namespace used within ``cache``.

        Returns:
            Tensor with shape ``[B, T, H, D]``.
        """
        if input.dim() != 4:
            raise ValueError("input must have shape [B, T, H, D]")
        batch_size, num_rows, num_columns, channels = input.shape
        if train_size.shape != (batch_size,):
            raise ValueError("train_size must have shape [B]")
        if train_size.is_floating_point():
            raise ValueError("train_size must have an integer dtype")
        if train_size.device != input.device:
            raise ValueError("train_size and input must use the same device")

        # [B, T, H, D] -> [B * H, T, D].
        source = input.permute(0, 2, 1, 3).reshape(
            batch_size * num_columns,
            num_rows,
            channels,
        )
        attn_mask = None
        if cache is None or cache.is_recording:
            expanded_train_size = train_size.repeat_interleave(num_columns)
            row_index = torch.arange(num_rows, device=input.device)
            attn_mask = row_index.unsqueeze(0) < expanded_train_size.unsqueeze(
                1
            )
            attn_mask = attn_mask[:, None, None, :]  # [B * H, 1, 1, T].

        chunk_size = self.col_chunk_size
        if (
            cache is not None
            and chunk_size is not None
            and source.size(0) > chunk_size
        ):
            if chunk_size <= 0:
                raise ValueError("col_chunk_size must be positive or None")
            output = self._cached_chunked_stage(
                source,
                attn_mask,
                cache=cache,
                cache_prefix=cache_prefix,
                chunk_size=chunk_size,
            )
        elif cache is not None:
            output = self._stage(
                source,
                attn_mask,
                cache=cache,
                cache_prefix=cache_prefix,
            )
        elif chunk_size is None or source.size(0) <= chunk_size:
            assert attn_mask is not None
            output = self._stage(source, attn_mask)
        else:
            if chunk_size <= 0:
                raise ValueError("col_chunk_size must be positive or None")
            assert attn_mask is not None
            # Columns are independent, so chunking bounds activation memory
            # without changing the attention context of any output.
            output = torch.cat(
                [
                    self._stage(
                        source[start : start + chunk_size],
                        attn_mask[start : start + chunk_size],
                    )
                    for start in range(0, source.size(0), chunk_size)
                ],
                dim=0,
            )

        # [B * H, T, D] -> [B, T, H, D].
        return output.reshape(
            batch_size,
            num_columns,
            num_rows,
            channels,
        ).permute(0, 2, 1, 3)
