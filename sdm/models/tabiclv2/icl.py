# ruff: noqa: D101, D102

import math
from typing import Any, TypeAlias, cast

import torch
from torch import Tensor
from torch.nn import GELU, Embedding, LayerNorm, Linear, ModuleList, Sequential

from sdm.cache import Cache, KVCacheEntry
from sdm.nn import TransformerBlock
from sdm.nn.memory import (
    attention_batch_size_limit,
    cuda_attention_memory_limit,
)

_Node: TypeAlias = dict[str, Tensor | list["_Node"]]


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
        self.num_heads = num_heads
        self.temperature = temperature

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
                norm="layer_norm",
                norm_kwargs={"bias": norm_bias},
                qassmax=True,
                **factory_kwargs,
            )
            self.layers.append(layer)

        self.norm = LayerNorm(channels, bias=norm_bias, **factory_kwargs)
        self.head = Sequential(
            Linear(
                in_features=channels,
                out_features=2 * channels,
                **factory_kwargs,
            ),
            GELU(),
            Linear(
                in_features=2 * channels,
                out_features=out_channels,
                **factory_kwargs,
            ),
        )

    def forward(
        self,
        x: Tensor,  # [..., R, D]
        y: Tensor,  # [..., R_train]
        *,
        num_classes: int | None = None,
        seqused_train: Tensor | None = None,  # [...]
        cache: Cache | None = None,
        batch_size_limit: int | None = None,
    ) -> Tensor:  # [..., R_test, out_channels or num_classes]
        if num_classes is None or num_classes <= self.num_classes:
            return self._forward(
                x=x,
                y=y,
                seqused_train=seqused_train,
                cache=cache,
                cache_prefix="icl_block",
                batch_size_limit=batch_size_limit,
            )

        if self.num_classes < 2:
            raise ValueError(
                "Hierarchical classification requires 'num_classes' to be "
                "at least two"
            )

        if seqused_train is not None:
            # Hierarchical nodes re-group the in-context rows by class, so
            # the "only the first `seqused_train` rows are valid" contract
            # no longer describes any node and padded rows would leak into
            # the class hierarchy.
            raise ValueError(
                "`seqused_train` padding is not supported for hierarchical "
                f"classification with more than {self.num_classes} classes"
            )

        return self._forward_hierarchical(
            x=x,
            y=y,
            num_classes=num_classes,
            cache=cache,
            batch_size_limit=batch_size_limit,
        )

    def _forward(
        self,
        x: Tensor,  # [..., R, D]
        y: Tensor,  # [..., R_train]
        *,
        seqused_train: Tensor | None,  # [...]
        cache: Cache | None,
        cache_prefix: str,
        batch_size_limit: int | None,
    ) -> Tensor:  # [..., R_test, out_channels]
        R_train = y.size(-1)

        if y.numel() > 0:
            if self.y_emb is not None:
                y_emb = self.y_emb(y)  # [..., R_train, D]
            else:
                assert self.y_lin is not None
                y_emb = self.y_lin(y.unsqueeze(-1))  # [..., R_train, D]

            x[..., :R_train, :] += y_emb.to(x.dtype)

        plan_attention = (
            x.device.type == "cuda"
            and not self.training
            and not torch.is_grad_enabled()
            and not torch.compiler.is_compiling()
        )

        icl_batch_size_limit = batch_size_limit
        for i, layer in enumerate(self.layers):
            key = f"{cache_prefix}.layer{i}"
            query = x[..., R_train:, :] if i == len(self.layers) - 1 else x
            key_value = (
                cast(KVCacheEntry, cache[key])
                if cache is not None and cache.is_replaying
                else x[..., :R_train, :]
            )
            if i == 0 or (
                plan_attention and cache is not None and cache.is_recording
            ):
                icl_batch_size_limit = attention_batch_size_limit(
                    requested_limit=batch_size_limit,
                    query=query,
                    key_value=key_value,
                    attention_memory_limit=(
                        cuda_attention_memory_limit(x.device)
                        if plan_attention
                        else None
                    ),
                    num_heads=self.num_heads,
                )
            result = layer(
                query=query,
                key_value=key_value,  # [..., R_train, D]
                seqused_key_value=seqused_train,  # [...]
                return_key_value=cache is not None and cache.is_recording,
                batch_size_limit=icl_batch_size_limit,
            )

            if cache is not None and cache.is_recording:
                x, cache[key] = result
            else:
                x = result

        return self.head(self.norm(x))  # [..., R_test, out_channels]

    def _forward_hierarchical(
        self,
        x: Tensor,  # [..., R, D]
        y: Tensor,  # [..., R_train]
        *,
        num_classes: int,
        cache: Cache | None,
        batch_size_limit: int | None,
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
            trees = cast(list[_Node], cache["icl_block.trees"])
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
                    batch_size_limit=batch_size_limit,
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
                    batch_size_limit=batch_size_limit,
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
        batch_size_limit: int | None,
    ) -> Tensor:  # [R_test, C]
        class_ids, local_log_probs = self._replay_node(
            test_rows=test_rows,
            node=node,
            cache=cache,
            cache_prefix=cache_prefix,
            batch_size_limit=batch_size_limit,
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
        batch_size_limit: int | None,
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
                    batch_size_limit=batch_size_limit,
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
            batch_size_limit=batch_size_limit,
        )

        child_class_ids: list[Tensor] = []
        children_log_probs: list[Tensor] = []
        children = cast(list[_Node], node["children"])
        for group_idx in range(num_groups):
            mask = group_masks[group_idx]
            child_ids, child_log_probs, child = self._process_node(
                train_rows=train_rows[mask],
                train_labels=train_labels[mask],
                test_rows=test_rows,
                cache=cache,
                cache_prefix=f"{cache_prefix}.child{group_idx}",
                batch_size_limit=batch_size_limit,
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
        batch_size_limit: int | None,
    ) -> tuple[Tensor, Tensor]:  # [C_node], [R_test, C_node]
        class_ids = cast(Tensor, node["class_ids"])
        children = cast(list[_Node], node["children"])

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
                    batch_size_limit=batch_size_limit,
                )
            return class_ids, local_log_probs

        group_log_probs = self._predict_log_probs(
            x=test_rows,
            y=class_ids.new_empty((0,)),
            num_classes=len(children),
            cache=cache,
            cache_prefix=cache_prefix,
            batch_size_limit=batch_size_limit,
        )
        child_class_ids: list[Tensor] = []
        children_log_probs: list[Tensor] = []
        for group_idx, child in enumerate(children):
            child_ids, child_log_probs = self._replay_node(
                test_rows=test_rows,
                node=child,
                cache=cache,
                cache_prefix=f"{cache_prefix}.child{group_idx}",
                batch_size_limit=batch_size_limit,
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
        batch_size_limit: int | None,
    ) -> Tensor:  # [R_test, C_node]
        logits = self._forward(
            x=x,
            y=y,
            # Hierarchical nodes hold exactly their own rows, so there is no
            # padding to mask (`forward` rejects `seqused_train` here).
            seqused_train=None,
            cache=cache,
            cache_prefix=cache_prefix,
            batch_size_limit=batch_size_limit,
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
