# Copyright (c) 2025, Soda team @ Inria
# Licensed under the BSD 3-Clause License; see LICENSE.

# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102

import math
from typing import Any, NamedTuple, TypeAlias, cast

import torch
from torch import Tensor
from torch.nn import GELU, Embedding, LayerNorm, Linear, ModuleList, Sequential

from sdm.cache import Cache, KVCacheEntry
from sdm.models.tabiclv2.block import TabICLv2TransformerBlock

_Node: TypeAlias = dict[str, Tensor | list["_Node"]]


class _HierarchyNode(NamedTuple):
    class_ids: Tensor
    row_indices: Tensor
    labels: Tensor
    children: tuple["_HierarchyNode", ...]


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
        chunk_memory_bytes: int | None = None,
        hierarchy: tuple[_HierarchyNode, ...] | None = None,
    ) -> Tensor:  # [..., R_test, out_channels or num_classes]
        if num_classes is None or num_classes <= self.num_classes:
            return self._forward(
                x=x,
                y=y,
                cache=cache,
                cache_prefix="icl_block",
                chunk_memory_bytes=chunk_memory_bytes,
            )

        if self.num_classes < 2:
            raise ValueError(
                "Hierarchical classification requires 'num_classes' to be "
                "at least two"
            )

        return self._forward_hierarchical(
            x=x,
            y=y,
            num_classes=num_classes,
            cache=cache,
            chunk_memory_bytes=chunk_memory_bytes,
            hierarchy=hierarchy,
        )

    def _forward(
        self,
        x: Tensor,  # [..., R, D]
        y: Tensor,  # [..., R_train]
        *,
        cache: Cache | None,
        cache_prefix: str,
        chunk_memory_bytes: int | None = None,
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
                return_key_value=cache is not None and cache.is_recording,
                chunk_memory_bytes=chunk_memory_bytes,
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

    def _forward_hierarchical(
        self,
        x: Tensor,  # [..., R, D]
        y: Tensor,  # [..., R_train]
        *,
        num_classes: int,
        cache: Cache | None,
        chunk_memory_bytes: int | None = None,
        hierarchy: tuple[_HierarchyNode, ...] | None = None,
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
                    chunk_memory_bytes=chunk_memory_bytes,
                )
                for table_idx, (rows, tree) in enumerate(zip(flat_rows, trees))
            ]
        elif hierarchy is not None:
            if len(hierarchy) != num_tables:
                raise RuntimeError(
                    "Prepared hierarchy does not match table count"
                )
            table_outputs = []
            for table_idx, (rows, node) in enumerate(
                zip(flat_rows, hierarchy)
            ):
                class_ids, local_log_probs = self._process_prepared_node(
                    train_rows=rows[:train_size],
                    test_rows=rows[train_size:],
                    node=node,
                    cache=cache,
                    cache_prefix=f"icl_block.table{table_idx}.node",
                    chunk_memory_bytes=chunk_memory_bytes,
                )
                table_outputs.append(
                    self._expand_log_probs(
                        class_ids=class_ids,
                        local_log_probs=local_log_probs,
                        num_classes=num_classes,
                    )
                )
            if cache is not None and cache.is_recording:
                cache["icl_block.trees"] = [
                    self._hierarchy_tree(node) for node in hierarchy
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
                    chunk_memory_bytes=chunk_memory_bytes,
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
        chunk_memory_bytes: int | None = None,
    ) -> Tensor:  # [R_test, C]
        class_ids, local_log_probs = self._replay_node(
            test_rows=test_rows,
            node=node,
            cache=cache,
            cache_prefix=cache_prefix,
            chunk_memory_bytes=chunk_memory_bytes,
        )
        return self._expand_log_probs(
            class_ids=class_ids,
            local_log_probs=local_log_probs,
            num_classes=num_classes,
        )

    def _prepare_hierarchy(
        self,
        y: Tensor,
        num_classes: int | None,
        cache: Cache | None = None,
    ) -> tuple[_HierarchyNode, ...] | None:
        """Prepare label-dependent topology outside the neural graph."""
        if num_classes is None or num_classes <= self.num_classes:
            return None
        if self.num_classes < 2:
            raise ValueError(
                "Hierarchical classification requires 'num_classes' to be "
                "at least two"
            )
        if cache is not None and cache.is_replaying:
            return None
        train_size = y.size(-1)
        num_tables = math.prod(y.shape[:-1])
        return tuple(
            self._prepare_hierarchy_node(
                labels,
                torch.arange(train_size, device=y.device),
            )
            for labels in y.reshape(num_tables, train_size)
        )

    def _prepare_hierarchy_node(
        self,
        labels: Tensor,
        row_indices: Tensor,
    ) -> _HierarchyNode:
        class_ids, local_labels = labels.unique(
            sorted=True, return_inverse=True
        )
        if class_ids.numel() <= self.num_classes:
            return _HierarchyNode(class_ids, row_indices, local_labels, ())
        assignments, num_groups = self._grouping(
            class_ids.numel(), labels.device
        )
        group_labels = assignments[local_labels]
        children = tuple(
            self._prepare_hierarchy_node(labels[mask], row_indices[mask])
            for group_idx in range(num_groups)
            for mask in (group_labels == group_idx,)
        )
        return _HierarchyNode(class_ids, row_indices, group_labels, children)

    @staticmethod
    def _hierarchy_tree(node: _HierarchyNode) -> _Node:
        return {
            "class_ids": node.class_ids,
            "children": [ICLBlock._hierarchy_tree(c) for c in node.children],
        }

    def _process_prepared_node(
        self,
        train_rows: Tensor,
        test_rows: Tensor,
        node: _HierarchyNode,
        *,
        cache: Cache | None,
        cache_prefix: str,
        chunk_memory_bytes: int | None = None,
    ) -> tuple[Tensor, Tensor]:
        node_num_classes = node.class_ids.numel()
        if node_num_classes == 1:
            return node.class_ids, test_rows.sum(dim=-1, keepdim=True).mul(0)
        local_log_probs = self._predict_log_probs(
            x=torch.cat(
                (train_rows.index_select(0, node.row_indices), test_rows)
            ),
            y=node.labels,
            num_classes=len(node.children)
            if node.children
            else node_num_classes,
            cache=cache,
            cache_prefix=cache_prefix,
            chunk_memory_bytes=chunk_memory_bytes,
        )
        if not node.children:
            return node.class_ids, local_log_probs
        child_ids = []
        child_outputs = []
        for group_idx, child in enumerate(node.children):
            ids, output = self._process_prepared_node(
                train_rows=train_rows,
                test_rows=test_rows,
                node=child,
                cache=cache,
                cache_prefix=f"{cache_prefix}.child{group_idx}",
                chunk_memory_bytes=chunk_memory_bytes,
            )
            child_ids.append(ids)
            child_outputs.append(
                output + local_log_probs[:, group_idx : group_idx + 1]
            )
        return torch.cat(child_ids), torch.cat(child_outputs, dim=-1)

    def _process_node(
        self,
        train_rows: Tensor,
        train_labels: Tensor,
        test_rows: Tensor,
        *,
        cache: Cache | None,
        cache_prefix: str,
        chunk_memory_bytes: int | None = None,
    ) -> tuple[Tensor, Tensor, _Node]:
        node = self._prepare_hierarchy_node(
            train_labels,
            torch.arange(train_labels.numel(), device=train_labels.device),
        )
        class_ids, log_probs = self._process_prepared_node(
            train_rows=train_rows,
            test_rows=test_rows,
            node=node,
            cache=cache,
            cache_prefix=cache_prefix,
            chunk_memory_bytes=chunk_memory_bytes,
        )
        return class_ids, log_probs, self._hierarchy_tree(node)

    def _replay_node(
        self,
        test_rows: Tensor,  # [R_test, D]
        node: _Node,
        *,
        cache: Cache,
        cache_prefix: str,
        chunk_memory_bytes: int | None = None,
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
                    chunk_memory_bytes=chunk_memory_bytes,
                )
            return class_ids, local_log_probs

        group_log_probs = self._predict_log_probs(
            x=test_rows,
            y=class_ids.new_empty((0,)),
            num_classes=len(children),
            cache=cache,
            cache_prefix=cache_prefix,
            chunk_memory_bytes=chunk_memory_bytes,
        )
        child_class_ids: list[Tensor] = []
        children_log_probs: list[Tensor] = []
        for group_idx, child in enumerate(children):
            child_ids, child_log_probs = self._replay_node(
                test_rows=test_rows,
                node=child,
                cache=cache,
                cache_prefix=f"{cache_prefix}.child{group_idx}",
                chunk_memory_bytes=chunk_memory_bytes,
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
        chunk_memory_bytes: int | None = None,
    ) -> Tensor:  # [R_test, C_node]
        logits = self._forward(
            x=x,
            y=y,
            cache=cache,
            cache_prefix=cache_prefix,
            chunk_memory_bytes=chunk_memory_bytes,
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
