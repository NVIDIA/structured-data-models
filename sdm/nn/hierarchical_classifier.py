"""Hierarchical classification for native class-limited predictors."""

import math
from collections.abc import Callable

import torch
from torch import Tensor

_Predictor = Callable[[Tensor, Tensor], Tensor]


class HierarchicalClassifier(torch.nn.Module):
    r"""Hierarchical classifier from the TabICLv2 model.

    Introduced in the `"TabICLv2: A Better, Faster, Scalable, and Open
    Tabular Foundation Model" <https://arxiv.org/abs/2602.11139>`_ paper, this
    module supports predictors with a fixed native class capacity.

    The classifier recursively partitions ordered class indices into balanced
    groups until every leaf fits within the native class capacity of the
    predictor. Each node receives only the training rows assigned to that node,
    while every test row is evaluated at every relevant node. Node
    probabilities are combined along the tree using the probability chain
    rule.

    Args:
        max_classes: Maximum number of classes supported natively by the
            predictor.
        temperature: Positive softmax temperature applied to predictor logits
            at every node.
    """

    def __init__(
        self,
        max_classes: int,
        temperature: float = 0.9,
    ) -> None:
        super().__init__()
        if max_classes < 1:
            raise ValueError("Expected 'max_classes' to be positive")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError(
                "Expected 'temperature' to be finite and positive"
            )

        self.max_classes = max_classes
        self.temperature = temperature

    def forward(
        self,
        row_embeddings: Tensor,  # [..., R, D]
        y: Tensor,  # [..., R_train]
        *,
        num_classes: int,
        predictor: _Predictor,
    ) -> Tensor:  # [..., R_test, C]
        """Predict class probabilities from row embeddings.

        Args:
            row_embeddings: Row embeddings with shape ``[..., R, D]``. The
                first ``R_train`` rows are the in-context training rows and the
                remaining ``R_test`` rows are test rows.
            y: Integer class indices in ``[0, num_classes)`` with shape
                ``[..., R_train]``. Classes absent from a table's context
                receive zero probability.
            num_classes: Number of classes ``C`` in the global output space.
            predictor: Callable receiving node rows with shape
                ``[R_node + R_test, D]`` and remapped node labels with shape
                ``[R_node]``. It must return logits with shape
                ``[R_test, max_classes]``.

        Returns:
            Class probabilities with shape ``[..., R_test, C]``.
        """
        if num_classes < 1:
            raise ValueError("Expected 'num_classes' to be positive")
        if y.numel() == 0:
            raise ValueError(
                "Expected at least one in-context classification label"
            )
        if y.is_floating_point() or y.is_complex():
            raise TypeError("Expected integer classification labels")

        y = y.long()
        *batch_shape, num_rows, channels = row_embeddings.size()
        train_size = y.size(-1)
        test_size = num_rows - train_size

        flat_rows = row_embeddings.reshape(-1, num_rows, channels)
        flat_y = y.reshape(-1, train_size)

        # Node contexts have different row counts, so tables and recursive
        # model calls cannot be represented by one dense tensor operation.
        probabilities = torch.stack(
            [
                self._process_node(
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

    def _process_node(
        self,
        train_rows: Tensor,  # [R_node, D]
        train_labels: Tensor,  # [R_node]
        test_rows: Tensor,  # [R_test, D]
        *,
        num_classes: int,
        predictor: _Predictor,
    ) -> Tensor:  # [R_test, C]
        class_ids, local_labels = train_labels.unique(
            sorted=True,
            return_inverse=True,
        )
        node_num_classes = class_ids.numel()
        test_size = test_rows.size(-2)

        if node_num_classes <= self.max_classes:
            if node_num_classes == 1:
                local_probs = test_rows.new_ones((test_size, 1))
            else:
                node_rows = torch.cat((train_rows, test_rows), dim=0)
                logits = predictor(node_rows, local_labels)
                local_probs = (
                    logits[..., :node_num_classes] / self.temperature
                ).softmax(dim=-1)

            global_probs = local_probs.new_zeros((test_size, num_classes))
            return global_probs.index_copy(-1, class_ids, local_probs)

        class_groups, num_groups = self._grouping(
            num_classes=node_num_classes,
            device=train_labels.device,
        )
        group_labels = class_groups[local_labels]
        node_rows = torch.cat((train_rows, test_rows), dim=0)
        group_logits = predictor(node_rows, group_labels)
        group_probs = (
            group_logits[..., :num_groups] / self.temperature
        ).softmax(dim=-1)

        final_probs = group_probs.new_zeros((test_size, num_classes))
        for group_idx in range(num_groups):
            mask = group_labels == group_idx
            child_probs = self._process_node(
                train_rows=train_rows[mask],
                train_labels=train_labels[mask],
                test_rows=test_rows,
                num_classes=num_classes,
                predictor=predictor,
            )
            final_probs = (
                final_probs
                + child_probs * group_probs[..., group_idx : group_idx + 1]
            )

        return final_probs

    def _grouping(
        self,
        num_classes: int,
        device: torch.device,
    ) -> tuple[Tensor, int]:
        if num_classes <= self.max_classes:
            assignments = torch.zeros(
                num_classes,
                dtype=torch.long,
                device=device,
            )
            return assignments, 1
        if self.max_classes < 2:
            raise ValueError(
                "Hierarchical classification requires at least two native "
                "classes"
            )

        num_groups = min(
            (num_classes + self.max_classes - 1) // self.max_classes,
            self.max_classes,
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
