import math
from collections.abc import Sequence
from typing import Literal

import torch

from sdm.processing._utils import _as_float
from sdm.processing.base import Processor
from sdm.processing.context import RecipeContext
from sdm.stype import Stype
from sdm.tensor import TableTensor

_RFM_QUANTILE_LEVELS = (
    0.005,
    0.01,
    0.02,
    0.025,
    0.05,
    0.1,
    0.15,
    0.2,
    0.25,
    0.3,
    0.35,
    0.4,
    0.45,
    0.5,
    0.55,
    0.6,
    0.65,
    0.7,
    0.75,
    0.8,
    0.85,
    0.9,
    0.95,
    0.975,
    0.98,
    0.99,
    0.995,
)


class SoftmaxTemperature(Processor):
    """Apply softmax to logits after temperature scaling.

    Args:
        temperature: Positive divisor applied to logits before softmax;
            higher values produce a softer distribution.
    """

    requires_fit = False

    supported_stypes = frozenset({Stype.numerical})

    def __init__(
        self,
        *,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive.")
        self.temperature = temperature

    def _transform(self, table: TableTensor) -> TableTensor:
        """Return ``softmax(table / temperature)`` over the last dimension."""
        numerical = torch.softmax(
            _as_float(table.numerical) / self.temperature,
            dim=-1,
        )
        return table.replace_blocks(numerical=numerical)


def _estimator_contexts(
    table: TableTensor,
    context: Sequence[RecipeContext] | None,
    *,
    processor: str,
    task: Literal["classification", "regression"],
) -> Sequence[RecipeContext]:
    assert context is not None
    if table.numerical.dim() < 3:
        raise ValueError(
            f"'{processor}' expects estimator, row, and output dimensions "
            f"(got shape {tuple(table.size())})."
        )

    num_estimators = table.numerical.size(-3)
    if len(context) != num_estimators:
        raise ValueError(
            f"'{processor}' received {len(context)} Recipe contexts for "
            f"{num_estimators} estimator outputs."
        )

    for estimator_index, estimator_context in enumerate(context):
        if estimator_context.estimator_index != estimator_index:
            raise ValueError(
                f"'{processor}' expected RecipeContext {estimator_index} at "
                f"position {estimator_index} (got "
                f"{estimator_context.estimator_index})."
            )
        if estimator_context.task != task:
            raise ValueError(f"'{processor}' requires {task} Recipe contexts.")
    return context


class ClassDecode(Processor):
    """Restore all estimator logits to the original target class order.

    The input shape is ``[..., estimators, rows, model outputs]``.
    ``class_indices=None`` is an explicit identity mapping for raw targets
    without class metadata.
    """

    supported_stypes = frozenset({Stype.numerical})
    required_context = frozenset({"task"})
    requires_fit = False

    def _transform(self, table: TableTensor) -> TableTensor:
        raise RuntimeError("ClassDecode requires Recipe contexts.")

    def _transform_with_context(
        self,
        table: TableTensor,
        context: Sequence[RecipeContext] | None,
    ) -> TableTensor:
        contexts = _estimator_contexts(
            table,
            context,
            processor=self.__class__.__name__,
            task="classification",
        )
        mappings = tuple(item.class_indices for item in contexts)
        if all(mapping is None for mapping in mappings):
            return table
        if any(mapping is None for mapping in mappings):
            raise ValueError(
                "ClassDecode received inconsistent class mappings."
            )

        class_indices = tuple(
            mapping for mapping in mappings if mapping is not None
        )
        num_classes = len(class_indices[0])
        if any(len(mapping) != num_classes for mapping in class_indices):
            raise ValueError(
                "ClassDecode requires the same class count for every "
                "estimator."
            )

        required_columns = (
            max(
                (index for mapping in class_indices for index in mapping),
                default=-1,
            )
            + 1
        )
        if table.numerical.size(-1) < required_columns:
            raise ValueError(
                "Expected classification output to contain at least "
                f"{required_columns} columns "
                f"(got {table.numerical.size(-1)})."
            )

        indices = torch.tensor(
            class_indices,
            device=table.device,
        )
        batch_shape = table.numerical.size()[:-3]
        indices = indices.reshape(
            *((1,) * len(batch_shape)),
            len(contexts),
            1,
            num_classes,
        )
        indices = indices.expand(
            *batch_shape,
            len(contexts),
            table.numerical.size(-2),
            num_classes,
        )
        numerical = table.numerical.gather(-1, indices)
        return TableTensor.from_tensor(numerical)


class TargetDecode(Processor):
    """Map every estimator regression output to original target space."""

    supported_stypes = frozenset({Stype.numerical})
    required_context = frozenset({"target_inverse", "task"})
    requires_fit = False

    def _transform(self, table: TableTensor) -> TableTensor:
        raise RuntimeError("TargetDecode requires Recipe contexts.")

    def _transform_with_context(
        self,
        table: TableTensor,
        context: Sequence[RecipeContext] | None,
    ) -> TableTensor:
        contexts = _estimator_contexts(
            table,
            context,
            processor=self.__class__.__name__,
            task="regression",
        )
        columns = table.columns[Stype.numerical]
        members = []
        for estimator_index, estimator_context in enumerate(contexts):
            member = TableTensor.from_tensor(
                table.numerical.select(-3, estimator_index),
                columns=columns,
            )
            assert estimator_context.target_inverse is not None
            member = estimator_context.target_inverse.inverse_transform(member)
            members.append(member.numerical)

        numerical = torch.stack(members, dim=-3)
        return TableTensor.from_tensor(numerical, columns=columns)


class EstimatorMean(Processor):
    """Average canonical outputs over the explicit estimator dimension.

    The input shape is ``[..., estimators, rows, outputs]``; the estimator
    dimension is therefore the third-to-last dimension.
    """

    supported_stypes = frozenset({Stype.numerical})
    requires_fit = False

    def _transform(self, table: TableTensor) -> TableTensor:
        if table.numerical.dim() < 3:
            raise ValueError(
                "EstimatorMean expects estimator, row, and output dimensions "
                f"(got shape {tuple(table.size())})."
            )
        numerical = table.numerical.mean(dim=-3)
        return TableTensor.from_tensor(
            numerical,
            columns=table.columns[Stype.numerical],
        )


class QuantileDecode(Processor):
    """Decode an ordered regression head into RFM-style outputs.

    Raw coordinates are sorted before selecting the requested statistic,
    matching KumoRFM v2.1 output handling.

    Args:
        method: Return the distribution mean, median, or selected quantiles.
        quantiles: Levels in ``[0, 1]`` used by ``method="quantiles"``.
    """

    supported_stypes = frozenset({Stype.numerical})
    requires_fit = False

    def __init__(
        self,
        *,
        method: Literal["mean", "median", "quantiles"] = "median",
        quantiles: Sequence[float] | None = None,
    ) -> None:
        super().__init__()
        if method not in {"mean", "median", "quantiles"}:
            raise ValueError("method must be 'mean', 'median', or 'quantiles'")
        if method == "quantiles":
            if quantiles is None:
                quantiles = _RFM_QUANTILE_LEVELS
            if len(quantiles) == 0:
                raise ValueError("quantiles must not be empty")
            if any(level < 0 or level > 1 for level in quantiles):
                raise ValueError("quantiles must be between 0 and 1")
        elif quantiles is not None:
            raise ValueError(
                "quantiles is only supported with method='quantiles'"
            )
        self.method = method
        self.quantiles = None if quantiles is None else tuple(quantiles)

    def _transform(self, table: TableTensor) -> TableTensor:
        ordered = table.numerical.sort(dim=-1).values
        num_quantiles = ordered.size(-1)
        if num_quantiles <= 1:
            raise ValueError(
                "QuantileDecode expects at least two model-output "
                f"coordinates (got {num_quantiles})."
            )

        if self.method == "mean":
            numerical = ordered.mean(dim=-1, keepdim=True)
            columns = ("mean",)
        elif self.method == "median":
            numerical = ordered[..., num_quantiles // 2].unsqueeze(-1)
            columns = ("median",)
        else:
            assert self.quantiles is not None
            indices = torch.tensor(
                [
                    int(
                        min(
                            num_quantiles - 1,
                            max(
                                0,
                                (num_quantiles - 1) * level + 0.5,
                            ),
                        )
                    )
                    for level in self.quantiles
                ],
                device=table.device,
            )
            numerical = ordered.index_select(-1, indices)
            columns = tuple(f"q{level:g}" for level in self.quantiles)

        return TableTensor.from_tensor(numerical, columns=columns)

    def __repr__(self, *, indent: int = 0) -> str:
        args = f"method='{self.method}'"
        if self.quantiles is not None:
            args += f", quantiles={self.quantiles}"
        return f"{' ' * indent}{self.__class__.__name__}({args})"
