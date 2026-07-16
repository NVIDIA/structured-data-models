# ruff: noqa: D101, D102

import math
from collections.abc import Callable

import torch
from torch import Tensor

_Predictor = Callable[[Tensor, Tensor], Tensor]


class HierarchicalClassifier(torch.nn.Module):
    def __init__(
        self,
        num_classes: int,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("Expected 'num_classes' to be at least two")
        self.num_classes = num_classes
        self.temperature = temperature

    def forward(
        self,
        row_embeddings: Tensor,  # [..., R, D]
        y: Tensor,  # [..., R_train]
        *,
        num_classes: int,
        predictor: _Predictor,
    ) -> Tensor:  # [..., R_test, C]
        y = y.long()

        *batch_shape, num_rows, channels = row_embeddings.size()
        train_size = y.size(-1)
        test_size = num_rows - train_size

        if 0 in batch_shape:
            empty = row_embeddings[..., train_size:, :].sum(
                dim=-1,
                keepdim=True,
            )
            return empty.expand(*batch_shape, test_size, num_classes)

        num_tables = math.prod(batch_shape)
        flat_rows = row_embeddings.reshape(num_tables, num_rows, channels)
        flat_y = y.reshape(num_tables, train_size)

        # Node contexts have different row counts, so tables and recursive
        # model calls cannot be represented by one dense tensor operation.
        probabilities = torch.stack(
            [
                self._predict_table(
                    train_rows=rows[:train_size],
                    train_labels=labels,
                    test_rows=rows[train_size:],
                    num_classes=num_classes,
                    predictor=predictor,
                )
                for rows, labels in zip(flat_rows, flat_y)
            ]
        )
        return probabilities.reshape(*batch_shape, test_size, num_classes)

    def _predict_table(
        self,
        train_rows: Tensor,  # [R_train, D]
        train_labels: Tensor,  # [R_train]
        test_rows: Tensor,  # [R_test, D]
        *,
        num_classes: int,
        predictor: _Predictor,
    ) -> Tensor:  # [R_test, C]
        class_ids, local_probs = self._process_node(
            train_rows=train_rows,
            train_labels=train_labels,
            test_rows=test_rows,
            predictor=predictor,
        )
        probabilities = local_probs.new_zeros(
            (test_rows.size(-2), num_classes)
        )
        return probabilities.index_copy(-1, class_ids, local_probs)

    def _process_node(
        self,
        train_rows: Tensor,  # [R_node, D]
        train_labels: Tensor,  # [R_node]
        test_rows: Tensor,  # [R_test, D]
        *,
        predictor: _Predictor,
    ) -> tuple[Tensor, Tensor]:  # [C_node], [R_test, C_node]
        class_ids, local_labels = train_labels.unique(
            sorted=True,
            return_inverse=True,
        )
        node_num_classes = class_ids.numel()

        if node_num_classes <= self.num_classes:
            if node_num_classes == 1:
                local_probs = test_rows.sum(dim=-1, keepdim=True).mul(0).add(1)
            else:
                local_probs = self._predict_probabilities(
                    train_rows=train_rows,
                    train_labels=local_labels,
                    test_rows=test_rows,
                    num_classes=node_num_classes,
                    predictor=predictor,
                )

            return class_ids, local_probs

        class_groups, num_groups = self._grouping(
            num_classes=node_num_classes,
            device=train_labels.device,
        )
        group_labels = class_groups[local_labels]
        group_masks = group_labels == torch.arange(
            num_groups,
            device=group_labels.device,
        ).unsqueeze(-1)  # [G, R_node]
        group_probs = self._predict_probabilities(
            train_rows=train_rows,
            train_labels=group_labels,
            test_rows=test_rows,
            num_classes=num_groups,
            predictor=predictor,
        )

        child_class_ids: list[Tensor] = []
        child_probabilities: list[Tensor] = []
        for group_idx in range(num_groups):
            mask = group_masks[group_idx]
            child_ids, child_probs = self._process_node(
                train_rows=train_rows[mask],
                train_labels=train_labels[mask],
                test_rows=test_rows,
                predictor=predictor,
            )
            child_class_ids.append(child_ids)
            child_probabilities.append(
                child_probs * group_probs[:, group_idx : group_idx + 1]
            )

        # Balanced groups are contiguous in sorted class order, so concatenated
        # child outputs retain the node's sorted class order.
        return (
            torch.cat(child_class_ids),
            torch.cat(child_probabilities, dim=-1),
        )

    def _predict_probabilities(
        self,
        train_rows: Tensor,  # [R_node, D]
        train_labels: Tensor,  # [R_node]
        test_rows: Tensor,  # [R_test, D]
        *,
        num_classes: int,
        predictor: _Predictor,
    ) -> Tensor:  # [R_test, C_node]
        rows = torch.cat((train_rows, test_rows), dim=0)
        logits = predictor(rows, train_labels)
        return (logits[:, :num_classes] / self.temperature).softmax(dim=-1)

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
