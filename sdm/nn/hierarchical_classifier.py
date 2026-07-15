import math
from collections.abc import Callable

import torch
from torch import Tensor

_Predictor = Callable[[Tensor, Tensor], Tensor]


class HierarchicalClassifier(torch.nn.Module):
    r"""Classifier from `TabICLv2 <https://arxiv.org/abs/2602.11139>`_.

    This module supports predictors with a fixed native class capacity.

    The classifier recursively partitions ordered class indices into balanced
    groups until every leaf fits within the native class capacity of the
    predictor. Each node receives only the training rows assigned to that node,
    while every test row is evaluated at every relevant node. Node
    probabilities are combined along the tree using the probability chain
    rule. Batched tables are processed independently because their tree nodes
    can contain different numbers of training rows.

    Args:
        max_classes: Maximum number of classes supported natively by the
            predictor. Must be at least two.
        temperature: Positive softmax temperature applied to predictor logits
            at every node.
    """

    def __init__(
        self,
        max_classes: int,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if not isinstance(max_classes, int):
            raise TypeError("Expected 'max_classes' to be an integer")
        if max_classes < 2:
            raise ValueError("Expected 'max_classes' to be at least two")
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
                ``[R_node]``. It must return floating-point logits with shape
                ``[R_test, C_node]``, where ``C_node`` may be larger than the
                number of classes at the node but must not be smaller.

        Returns:
            Class probabilities with shape ``[..., R_test, C]``. Their dtype
            follows the predictor logits, except that a context containing
            only one class uses the row embedding dtype.
        """
        if not isinstance(num_classes, int):
            raise TypeError("Expected 'num_classes' to be an integer")
        if num_classes < 1:
            raise ValueError("Expected 'num_classes' to be positive")
        if row_embeddings.dim() < 2:
            raise ValueError(
                "Expected 'row_embeddings' to have at least two dimensions"
            )
        if not row_embeddings.is_floating_point():
            raise TypeError("Expected floating-point row embeddings")
        if y.dim() < 1:
            raise ValueError("Expected 'y' to have at least one dimension")
        if row_embeddings.size()[:-2] != y.size()[:-1]:
            raise ValueError(
                "Expected 'row_embeddings' and 'y' to share batch "
                f"dimensions (got {tuple(row_embeddings.size()[:-2])} and "
                f"{tuple(y.size()[:-1])})"
            )
        if row_embeddings.device != y.device:
            raise ValueError(
                "Expected 'row_embeddings' and 'y' to be on the same device "
                f"(got {row_embeddings.device} and {y.device})"
            )
        if y.size(-1) == 0:
            raise ValueError(
                "Expected at least one in-context classification label"
            )
        if y.size(-1) > row_embeddings.size(-2):
            raise ValueError(
                "Expected no more labels than row embeddings "
                f"(got {y.size(-1)} labels and "
                f"{row_embeddings.size(-2)} row embeddings)"
            )
        if y.is_floating_point() or y.is_complex():
            raise TypeError("Expected integer classification labels")
        y = y.long()
        if not ((y >= 0) & (y < num_classes)).all():
            raise ValueError(
                f"Expected 'y' values to be in the interval [0, {num_classes})"
            )
        if not callable(predictor):
            raise TypeError("Expected 'predictor' to be callable")

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

        if node_num_classes <= self.max_classes:
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
        if not isinstance(logits, Tensor):
            raise TypeError("Expected 'predictor' to return a Tensor")
        if logits.dim() != 2:
            raise ValueError(
                "Expected 'predictor' logits to have two dimensions "
                f"(got shape {tuple(logits.size())})"
            )
        if logits.size(0) != test_rows.size(0):
            raise ValueError(
                "Expected 'predictor' to return one logit row per test row "
                f"(got {logits.size(0)} and {test_rows.size(0)})"
            )
        if logits.size(1) < num_classes:
            raise ValueError(
                "Expected 'predictor' to return at least one logit per node "
                f"class (got {logits.size(1)} logits and {num_classes} "
                "classes)"
            )
        if not logits.is_floating_point():
            raise TypeError(
                "Expected 'predictor' to return floating-point logits"
            )
        if logits.device != test_rows.device:
            raise ValueError(
                "Expected 'predictor' logits and row embeddings to be on the "
                f"same device (got {logits.device} and {test_rows.device})"
            )

        return (logits[:, :num_classes] / self.temperature).softmax(dim=-1)

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
        assignments = group_indices.repeat_interleave(
            group_sizes,
            output_size=num_classes,
        )
        return assignments, num_groups
