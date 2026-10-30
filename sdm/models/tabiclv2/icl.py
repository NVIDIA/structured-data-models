# Copyright (c) 2025, Soda team @ Inria
# Licensed under the BSD 3-Clause License; see LICENSE.

# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102

import math
from typing import Any, TypeAlias, cast

import torch
from torch import Tensor
from torch.nn import GELU, Embedding, LayerNorm, Linear, ModuleList, Sequential

from sdm.cache import Cache, KVCacheEntry
from sdm.models.tabiclv2.block import TabICLv2TransformerBlock

_Node: TypeAlias = dict[str, Tensor | list["_Node"]]
_Level: TypeAlias = tuple[Tensor, Tensor, Tensor, Tensor]


class ICLBlock(torch.nn.Module):
    def __init__(
        self,
        num_classes: int,
        out_channels: int,
        channels: int,
        num_layers: int,
        num_heads: int,
        norm_bias: bool,
        temperature: float = 1.0,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}

        self.num_classes = num_classes
        self.temperature = temperature

        self.y_emb: torch.nn.Module | None = None
        self.y_lin: torch.nn.Module | None = None
        if num_classes > 0:
            self.y_emb = Embedding(num_classes, channels, **factory_kwargs)
        else:
            self.y_lin = Linear(1, channels, **factory_kwargs)

        self.layers = ModuleList(
            TabICLv2TransformerBlock(
                channels=channels,
                num_heads=num_heads,
                norm_bias=norm_bias,
                qassmax=True,
                **factory_kwargs,
            )
            for _ in range(num_layers)
        )

        self.norm = LayerNorm(channels, bias=norm_bias, **factory_kwargs)
        self.head = Sequential(
            Linear(channels, 2 * channels, **factory_kwargs),
            GELU(),
            Linear(2 * channels, out_channels, **factory_kwargs),
        )

    def forward(
        self,
        x: Tensor,  # [..., R, D]
        y: Tensor,  # [..., R_train]
        *,
        num_classes: int | None = None,
        cache: Cache | None = None,
    ) -> Tensor:  # [..., R_test, out_channels or num_classes]
        if (
            cache is not None
            and cache.is_replaying
            and "icl_block.levels" in cache
        ):
            levels = cast("list[_Level]", cache["icl_block.levels"])
            if levels[0][0].size(-1) != num_classes:
                raise RuntimeError(
                    "Cached hierarchy class count does not match"
                )

        if num_classes is None or num_classes <= self.num_classes:
            return self._forward(
                x=x,
                y=y,
                cache=cache,
                cache_prefix="icl_block",
            )

        if self.num_classes < 2:
            raise ValueError(
                "Hierarchical classification requires 'num_classes' to be "
                "at least two"
            )

        # Compilation needs fixed shapes; eager fit can keep compact caches.
        # Either cache layout supports eager and compiled prediction.
        if (
            torch.compiler.is_compiling()
            and (cache is None or cache.is_recording)
        ) or (cache is not None and "icl_block.levels" in cache):
            return self._forward_hierarchical_masked(
                x=x, y=y, num_classes=num_classes, cache=cache
            )

        return self._forward_hierarchical(
            x=x,
            y=y,
            num_classes=num_classes,
            cache=cache,
        )

    def _forward(
        self,
        x: Tensor,  # [..., R, D]
        y: Tensor,  # [..., R_train]
        *,
        cache: Cache | None,
        cache_prefix: str,
        attn_mask: Tensor | None = None,
    ) -> Tensor:  # [..., R_test, out_channels]
        R_train = y.size(-1)

        if y.numel() > 0:
            if self.y_emb is not None:
                y_emb = self.y_emb(y)  # [..., R_train, D]
            else:
                assert self.y_lin is not None
                y_emb = self.y_lin(y.unsqueeze(-1))  # [..., R_train, D]

            x[..., :R_train, :] += y_emb.to(x.dtype)

        for i, layer in enumerate(self.layers):
            key = f"{cache_prefix}.layer{i}"
            result = layer(
                query=x[..., R_train:, :] if i == len(self.layers) - 1 else x,
                key_value=(
                    cast(KVCacheEntry, cache[key])
                    if cache is not None and cache.is_replaying
                    else x[..., :R_train, :]
                ),
                attn_mask=attn_mask,
                return_key_value=cache is not None and cache.is_recording,
                # `x` is still the caller's tensor at i == 0; don't mutate.
                out=None
                if torch.is_grad_enabled() or i == 0
                else x[..., R_train:, :]
                if i == len(self.layers) - 1
                else x,
            )

            if cache is not None and cache.is_recording:
                x, cache[key] = result
            else:
                x = result
            del result

        return self.head(self.norm(x))  # [..., R_test, out_channels]

    def _forward_hierarchical_masked(
        self,
        x: Tensor,  # [..., R, D]
        y: Tensor,  # [..., R_train]
        *,
        num_classes: int,
        cache: Cache | None,
    ) -> Tensor:  # [..., R_test, C]
        y = y.long()
        train_size = y.size(-1)
        test_size = x.size(-2) - train_size
        if 0 in x.size()[:-2]:
            return (
                x[..., train_size:, :]
                .sum(dim=-1, keepdim=True)
                .expand(*x.size()[:-2], test_size, num_classes)
            )

        if cache is not None and cache.is_replaying:
            levels = cast("list[_Level]", cache["icl_block.levels"])
            if levels[0][0].size()[:-2] != x.size()[:-2]:
                raise RuntimeError(
                    "Cached hierarchy batch shape does not match"
                )
        else:
            levels = self._hierarchy(y=y, num_classes=num_classes)
            if cache is not None and cache.is_recording:
                cache["icl_block.levels"] = levels

        # Start at the input precision so autocast cannot make the output
        # dtype depend on which nodes are singletons.
        log_probs = x.new_zeros((*x.size()[:-2], test_size, num_classes))
        output_ids = torch.arange(self.num_classes, device=x.device)
        # Add child probabilities before parent probabilities, preserving the
        # recursive evaluation order in reduced precision.
        for depth in range(len(levels) - 1, -1, -1):
            classes, indices, sizes, mask = levels[depth]
            num_nodes = classes.size(-2)
            # Every level batches its nodes. Masking context rows keeps shapes
            # independent of label values, including missing classes.
            rows = (
                x.unsqueeze(-3)
                .expand(*x.size()[:-2], num_nodes, *x.size()[-2:])
                .clone()
            )
            labels = indices.gather(
                -1, y.unsqueeze(-2).expand(*y.size()[:-1], num_nodes, -1)
            )
            logits = self._forward(
                x=rows,
                y=labels,
                cache=cache,
                cache_prefix=f"icl_block.level{depth}",
                attn_mask=mask.unsqueeze(-2),
            )
            logits = (
                logits[..., : self.num_classes] / self.temperature
            ).masked_fill(output_ids >= sizes[..., None, None], -torch.inf)
            node_log_probs = logits.log_softmax(dim=-1)
            class_log_probs = node_log_probs.gather(
                -1,
                indices.unsqueeze(-2).expand(
                    *indices.size()[:-1], test_size, -1
                ),
            )
            log_probs = log_probs + class_log_probs.masked_fill(
                ~classes.unsqueeze(-2), 0
            ).sum(dim=-3)

        present = levels[0][0].squeeze(-2)
        # The output Softmax divides by temperature.
        return log_probs.masked_fill(~present.unsqueeze(-2), -torch.inf).mul(
            self.temperature
        )

    def _hierarchy(
        self,
        y: Tensor,  # [..., R_train]
        *,
        num_classes: int,
    ) -> list[_Level]:
        counts = y.new_zeros((*y.size()[:-1], num_classes))
        counts = counts.scatter_add(-1, y, torch.ones_like(y))
        classes = counts.gt(0).unsqueeze(-2)  # [..., N, C]
        levels = []
        bounds = [min(num_classes, y.size(-1))]
        while True:
            sizes = classes.sum(dim=-1, keepdim=True)
            ranks = (classes.cumsum(dim=-1) - 1).clamp(min=0)
            groups = (
                (sizes + self.num_classes - 1) // self.num_classes
            ).clamp(min=1, max=self.num_classes)
            quotient = sizes // groups
            remainder = sizes % groups
            boundary = (quotient + 1) * remainder
            # Balanced, contiguous groups of *present* classes, with larger
            # groups first, matching tensor_split in sorted class order.
            indices = torch.where(
                ranks < boundary,
                ranks // (quotient + 1),
                remainder + (ranks - boundary) // quotient.clamp(min=1),
            )
            internal = sizes > self.num_classes
            indices = torch.where(internal, indices, ranks).masked_fill(
                ~classes, 0
            )
            output_sizes = torch.where(internal, groups, sizes).squeeze(-1)
            mask = classes.gather(
                -1,
                y.unsqueeze(-2).expand(*y.size()[:-1], classes.size(-2), -1),
            )
            # Empty nodes need a finite softmax before masking their outputs.
            levels.append((classes, indices, output_sizes.clamp(min=1), mask))
            # Upper bounds omit impossible descendants. For 101 classes at
            # capacity 10, only the 11-class group splits again.
            parents, children, next_bounds = [], [], []
            for node, bound in enumerate(bounds):
                if bound <= self.num_classes:
                    continue
                max_groups = min(
                    (bound + self.num_classes - 1) // self.num_classes,
                    self.num_classes,
                )
                for child in range(max_groups):
                    parents.append(node)
                    children.append(child)
                    next_bounds.append(
                        max(
                            self.num_classes,
                            (bound + self.num_classes - 1 - child)
                            // self.num_classes,
                        )
                    )
            if not parents:
                return levels
            parent_ids = torch.tensor(parents, device=y.device)
            child_ids = torch.tensor(children, device=y.device)
            classes = (classes & internal).index_select(-2, parent_ids) & (
                indices.index_select(-2, parent_ids) == child_ids.unsqueeze(-1)
            )
            bounds = next_bounds

    def _forward_hierarchical(
        self,
        x: Tensor,  # [..., R, D]
        y: Tensor,  # [..., R_train]
        *,
        num_classes: int,
        cache: Cache | None,
    ) -> Tensor:  # [..., R_test, C]
        *batch_shape, num_rows, channels = x.size()
        train_size = y.size(-1)
        test_size = num_rows - train_size

        if 0 in batch_shape:
            empty = x[..., train_size:, :].sum(dim=-1, keepdim=True)
            return empty.expand(*batch_shape, test_size, num_classes)

        num_tables = math.prod(batch_shape)
        flat_rows = x.reshape(num_tables, num_rows, channels)

        # Tree nodes have different row counts, so tables and recursive model
        # calls cannot be represented by one dense tensor operation.
        if cache is not None and cache.is_replaying:
            trees = cast("list[_Node]", cache["icl_block.trees"])
            if len(trees) != num_tables:
                raise RuntimeError(
                    f"Expected {len(trees)} cached tables (got {num_tables})"
                )
            table_outputs = [
                self._replay_table(
                    test_rows=rows[train_size:],
                    node=tree,
                    num_classes=num_classes,
                    cache=cache,
                    cache_prefix=f"icl_block.table{table_idx}.node",
                )
                for table_idx, (rows, tree) in enumerate(zip(flat_rows, trees))
            ]
        else:
            flat_y = y.reshape(num_tables, train_size)
            trees = []
            table_outputs = []
            for table_idx, (rows, labels) in enumerate(zip(flat_rows, flat_y)):
                class_ids, local_log_probs, tree = self._process_node(
                    train_rows=rows[:train_size],
                    train_labels=labels,
                    test_rows=rows[train_size:],
                    cache=cache,
                    cache_prefix=f"icl_block.table{table_idx}.node",
                )
                trees.append(tree)
                table_outputs.append(
                    self._expand_log_probs(
                        class_ids=class_ids,
                        local_log_probs=local_log_probs,
                        num_classes=num_classes,
                    )
                )

            if cache is not None and cache.is_recording:
                cache["icl_block.trees"] = trees

        log_probs = torch.stack(table_outputs)
        # The output Softmax divides by temperature, so scale the combined
        # log-probabilities to preserve their normalized probabilities.
        return log_probs.reshape(*batch_shape, test_size, num_classes).mul(
            self.temperature
        )

    def _replay_table(
        self,
        test_rows: Tensor,  # [R_test, D]
        node: _Node,
        *,
        num_classes: int,
        cache: Cache,
        cache_prefix: str,
    ) -> Tensor:  # [R_test, C]
        class_ids, local_log_probs = self._replay_node(
            test_rows=test_rows,
            node=node,
            cache=cache,
            cache_prefix=cache_prefix,
        )
        return self._expand_log_probs(
            class_ids=class_ids,
            local_log_probs=local_log_probs,
            num_classes=num_classes,
        )

    def _process_node(
        self,
        train_rows: Tensor,  # [R_node, D]
        train_labels: Tensor,  # [R_node]
        test_rows: Tensor,  # [R_test, D]
        *,
        cache: Cache | None,
        cache_prefix: str,
    ) -> tuple[Tensor, Tensor, _Node]:  # [C_node], [R_test, C_node]
        class_ids, local_labels = train_labels.unique(
            sorted=True,
            return_inverse=True,
        )
        node_num_classes = class_ids.numel()
        node: _Node = {"class_ids": class_ids, "children": []}

        if node_num_classes <= self.num_classes:
            if node_num_classes == 1:
                local_log_probs = test_rows.sum(
                    dim=-1,
                    keepdim=True,
                ).mul(0)
            else:
                local_log_probs = self._predict_log_probs(
                    x=torch.cat((train_rows, test_rows), dim=0),
                    y=local_labels,
                    num_classes=node_num_classes,
                    cache=cache,
                    cache_prefix=cache_prefix,
                )

            return class_ids, local_log_probs, node

        class_groups, num_groups = self._grouping(
            num_classes=node_num_classes,
            device=train_labels.device,
        )
        group_labels = class_groups[local_labels]
        group_masks = group_labels == torch.arange(
            num_groups,
            device=group_labels.device,
        ).unsqueeze(-1)  # [G, R_node]
        group_log_probs = self._predict_log_probs(
            x=torch.cat((train_rows, test_rows), dim=0),
            y=group_labels,
            num_classes=num_groups,
            cache=cache,
            cache_prefix=cache_prefix,
        )

        child_class_ids: list[Tensor] = []
        children_log_probs: list[Tensor] = []
        children = cast("list[_Node]", node["children"])
        for group_idx in range(num_groups):
            mask = group_masks[group_idx]
            child_ids, child_log_probs, child = self._process_node(
                train_rows=train_rows[mask],
                train_labels=train_labels[mask],
                test_rows=test_rows,
                cache=cache,
                cache_prefix=f"{cache_prefix}.child{group_idx}",
            )
            child_class_ids.append(child_ids)
            children_log_probs.append(
                child_log_probs + group_log_probs[:, group_idx : group_idx + 1]
            )
            children.append(child)

        # Balanced groups are contiguous in sorted class order, so concatenated
        # child outputs retain the node's sorted class order.
        return (
            torch.cat(child_class_ids),
            torch.cat(children_log_probs, dim=-1),
            node,
        )

    def _replay_node(
        self,
        test_rows: Tensor,  # [R_test, D]
        node: _Node,
        *,
        cache: Cache,
        cache_prefix: str,
    ) -> tuple[Tensor, Tensor]:  # [C_node], [R_test, C_node]
        class_ids = cast(Tensor, node["class_ids"])
        children = cast("list[_Node]", node["children"])

        if not children:
            if class_ids.numel() == 1:
                local_log_probs = test_rows.sum(
                    dim=-1,
                    keepdim=True,
                ).mul(0)
            else:
                local_log_probs = self._predict_log_probs(
                    x=test_rows,
                    y=class_ids.new_empty((0,)),
                    num_classes=class_ids.numel(),
                    cache=cache,
                    cache_prefix=cache_prefix,
                )
            return class_ids, local_log_probs

        group_log_probs = self._predict_log_probs(
            x=test_rows,
            y=class_ids.new_empty((0,)),
            num_classes=len(children),
            cache=cache,
            cache_prefix=cache_prefix,
        )
        child_class_ids: list[Tensor] = []
        children_log_probs: list[Tensor] = []
        for group_idx, child in enumerate(children):
            child_ids, child_log_probs = self._replay_node(
                test_rows=test_rows,
                node=child,
                cache=cache,
                cache_prefix=f"{cache_prefix}.child{group_idx}",
            )
            child_class_ids.append(child_ids)
            children_log_probs.append(
                child_log_probs + group_log_probs[:, group_idx : group_idx + 1]
            )

        return (
            torch.cat(child_class_ids),
            torch.cat(children_log_probs, dim=-1),
        )

    def _predict_log_probs(
        self,
        x: Tensor,  # [R_node + R_test, D] or [R_test, D] with cache
        y: Tensor,  # [R_node] or [0] with cache
        *,
        num_classes: int,
        cache: Cache | None,
        cache_prefix: str,
    ) -> Tensor:  # [R_test, C_node]
        logits = self._forward(
            x=x,
            y=y,
            cache=cache,
            cache_prefix=cache_prefix,
        )
        return (logits[:, :num_classes] / self.temperature).log_softmax(dim=-1)

    @staticmethod
    def _expand_log_probs(
        class_ids: Tensor,  # [C_node]
        local_log_probs: Tensor,  # [R_test, C_node]
        *,
        num_classes: int,
    ) -> Tensor:  # [R_test, C]
        log_probs = local_log_probs.new_full(
            (local_log_probs.size(-2), num_classes),
            -torch.inf,
        )
        return log_probs.index_copy(
            -1,
            class_ids.to(torch.long),
            local_log_probs,
        )

    def _grouping(
        self,
        num_classes: int,
        device: torch.device,
    ) -> tuple[Tensor, int]:
        if num_classes <= self.num_classes:
            assignments = torch.zeros(
                num_classes,
                dtype=torch.long,
                device=device,
            )
            return assignments, 1
        num_groups = min(
            (num_classes + self.num_classes - 1) // self.num_classes,
            self.num_classes,
        )
        group_sizes = torch.full(
            (num_groups,),
            num_classes // num_groups,
            dtype=torch.long,
            device=device,
        )
        group_sizes[: num_classes % num_groups] += 1
        group_indices = torch.arange(num_groups, device=device)
        assignments = group_indices.repeat_interleave(
            group_sizes,
            output_size=num_classes,
        )
        return assignments, num_groups
