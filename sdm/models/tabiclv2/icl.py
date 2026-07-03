# ruff: noqa: D101, D102

from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor
from torch.nn import Embedding, LayerNorm, Linear, ModuleList

from sdm.cache import Cache
from sdm.nn import TransformerBlock


class ICLBlock(torch.nn.Module):
    def __init__(
        self,
        num_classes: int,
        channels: int,
        num_layers: int,
        num_heads: int,
        norm_bias: bool,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}

        self.y_emb: torch.nn.Module | None = None
        self.y_lin: torch.nn.Module | None = None
        if num_classes > 0:
            self.y_emb = Embedding(num_classes, channels, **factory_kwargs)
        else:
            self.y_lin = Linear(1, channels, **factory_kwargs)

        self.layers = ModuleList()
        for _ in range(num_layers):
            layer = TransformerBlock(
                channels=channels,
                num_query_heads=num_heads,
                feedforward_channels=2 * channels,
                qassmax=True,
                norm_bias=norm_bias,
                **factory_kwargs,
            )
            self.layers.append(layer)

        self.norm = LayerNorm(channels, bias=norm_bias, **factory_kwargs)

    def forward(
        self,
        x: Tensor,  # [..., R, D]
        y: Tensor,  # [..., R_train]
        *,
        cache: Cache | None = None,
    ) -> Tensor:  # [..., R_test, D]
        R_train = y.size(-1)

        if y.numel() > 0:
            if self.y_emb is not None:
                y_emb = self.y_emb(y)  # [..., R_train, D]
            else:
                assert self.y_lin is not None
                y_emb = self.y_lin(y.unsqueeze(-1))  # [..., R_train, D]

            x[..., :R_train, :] += y_emb.to(x.dtype)

        for i, layer in enumerate(self.layers):
            key = f"icl_block.layer{i}"
            result = layer(
                query=x[..., R_train:, :] if i == len(self.layers) - 1 else x,
                key_value=cache[key]
                if cache is not None and cache.is_replaying
                else x[..., :R_train, :],  # [..., R_train, D]
                return_key_value=cache is not None and cache.is_recording,
            )

            if cache is not None and cache.is_recording:
                x, cache[key] = result
            else:
                x = result

        return self.norm(x)  # [..., R_test, D]


def _balanced_grouping(
    num_classes: int,
    max_classes: int,
    device: torch.device,
) -> tuple[Tensor, int]:
    if num_classes <= max_classes:
        return torch.zeros(num_classes, dtype=torch.long, device=device), 1
    if max_classes < 2:
        raise ValueError(
            "Hierarchical classification requires at least two native classes"
        )

    num_groups = min(
        (num_classes + max_classes - 1) // max_classes,
        max_classes,
    )
    group_sizes = torch.full(
        (num_groups,),
        num_classes // num_groups,
        dtype=torch.long,
        device=device,
    )
    group_sizes[: num_classes % num_groups] += 1
    group_indices = torch.arange(num_groups, device=device)
    return group_indices.repeat_interleave(group_sizes), num_groups


def _predict_hierarchical(
    row_embeddings: Tensor,  # [..., R, D]
    y: Tensor,  # [..., R_train]
    *,
    num_classes: int,
    max_classes: int,
    temperature: float,
    predictor: Callable[[Tensor, Tensor], Tensor],
) -> Tensor:  # [..., R_test, C]
    *batch_shape, num_rows, channels = row_embeddings.size()
    train_size = y.size(-1)
    test_size = num_rows - train_size

    def _process_node(
        train_rows: Tensor,  # [R_node, D]
        train_labels: Tensor,  # [R_node]
        test_rows: Tensor,  # [R_test, D]
    ) -> Tensor:  # [R_test, C]
        class_ids, local_labels = train_labels.unique(
            sorted=True,
            return_inverse=True,
        )
        node_num_classes = class_ids.numel()

        if node_num_classes <= max_classes:
            if node_num_classes == 1:
                local_probs = test_rows.new_ones((test_size, 1))
            else:
                node_rows = torch.cat((train_rows, test_rows), dim=0)
                logits = predictor(node_rows, local_labels)
                local_probs = (
                    logits[..., :node_num_classes] / temperature
                ).softmax(dim=-1)

            global_probs = local_probs.new_zeros((test_size, num_classes))
            return global_probs.index_copy(-1, class_ids, local_probs)

        class_groups, num_groups = _balanced_grouping(
            num_classes=node_num_classes,
            max_classes=max_classes,
            device=train_labels.device,
        )
        group_labels = class_groups[local_labels]
        node_rows = torch.cat((train_rows, test_rows), dim=0)
        group_logits = predictor(node_rows, group_labels)
        group_probs = (group_logits[..., :num_groups] / temperature).softmax(
            dim=-1
        )

        final_probs = group_probs.new_zeros((test_size, num_classes))
        # Node contexts have different row counts, so recursive model calls
        # cannot be represented by one dense, vectorized tensor operation.
        for group_idx in range(num_groups):
            mask = group_labels == group_idx
            child_probs = _process_node(
                train_rows=train_rows[mask],
                train_labels=train_labels[mask],
                test_rows=test_rows,
            )
            final_probs = (
                final_probs
                + child_probs * group_probs[..., group_idx : group_idx + 1]
            )

        return final_probs

    flat_rows = row_embeddings.reshape(-1, num_rows, channels)
    flat_y = y.reshape(-1, train_size)
    probabilities = torch.stack(
        [
            _process_node(
                train_rows=rows[:train_size],
                train_labels=labels,
                test_rows=rows[train_size:],
            )
            for rows, labels in zip(flat_rows, flat_y)
        ]
    )
    return probabilities.reshape(*batch_shape, test_size, num_classes)
