"""Relative-time feature encoding for KumoRFM."""

import torch
from torch import Tensor

from sdm import Stype, TableTensor
from sdm.models.kumorfm.graph import HomogeneousGraph
from sdm.processing import (
    Clip,
    ConstantFilter,
    MeanImpute,
    Sequential,
    StandardScale,
)
from sdm.processing.datetime import US_PER_DAY

RelativeTimeProcessors = tuple[Sequential | None, Sequential | None]


def task_anchor(table: TableTensor, task_time_column: str) -> Tensor:
    """Return the task timestamp used to anchor relative-time features."""
    if table.stype(task_time_column) != Stype.datetime:
        raise ValueError(
            f"Task time column '{task_time_column}' must be a datetime column"
        )
    return table[task_time_column].datetime.squeeze(-1)


def relative_days(anchor: Tensor, timestamps: Tensor) -> Tensor:
    """Return ``anchor - timestamps`` in days without lossy timestamp casts."""
    missing = torch.iinfo(timestamps.dtype).min
    valid = (anchor != missing).unsqueeze(-1) & (timestamps != missing)
    safe_anchor = torch.where(anchor == missing, 0, anchor).unsqueeze(-1)
    safe_timestamps = torch.where(valid, timestamps, safe_anchor)

    relative = (
        safe_anchor.div(US_PER_DAY, rounding_mode="floor")
        - safe_timestamps.div(US_PER_DAY, rounding_mode="floor")
    ).to(torch.float32)
    relative += (
        safe_anchor.remainder(US_PER_DAY)
        - safe_timestamps.remainder(US_PER_DAY)
    ).to(torch.float32) / US_PER_DAY
    return relative.masked_fill(~valid, float("nan"))


def propagate_anchor_indices(
    num_anchors: int,
    root_index: Tensor,
    graph: HomogeneousGraph,
    num_hops: int,
) -> Tensor:
    """Propagate each task anchor index from its entity-table root."""
    state = torch.full(
        (graph.colptr.numel() - 1,),
        -1.0,
        device=root_index.device,
    )
    state.index_copy_(
        0,
        root_index,
        torch.arange(num_anchors, device=root_index.device).to(state),
    )

    for _ in range(num_hops):
        neighbor = torch.segment_reduce(
            state[graph.row],
            offsets=graph.colptr,
            reduce="max",
            unsafe=True,
            initial=-1,
        )
        state = torch.where(state >= 0, state, neighbor)

    return state.to(torch.long)


def relative_days_for_rows(
    anchor: Tensor,
    anchor_index: Tensor,
    timestamps: Tensor,
) -> Tensor:
    """Compute relative days for rows assigned to propagated task anchors."""
    relative = relative_days(
        anchor=anchor[anchor_index.clamp(min=0)],
        timestamps=timestamps,
    )
    return relative.masked_fill(anchor_index.unsqueeze(-1) < 0, float("nan"))


def fit_relative_time_features(
    context: Tensor,
    columns: tuple[str, ...],
    dtype: torch.dtype,
) -> tuple[Tensor, RelativeTimeProcessors]:
    """Fit relative-time preprocessing on context."""
    if len(columns) == 0:
        return context.to(dtype), (None, None)

    prepare = Sequential(
        MeanImpute(),
        ConstantFilter(method="variance", tolerance=0.0),
    )
    context_table = prepare.fit_transform(
        _relative_time_table(context, columns=columns)
    )
    if context_table.numerical.size(-1) > 0:
        normalize = Sequential(
            StandardScale(epsilon=1e-6),
            Clip(min_value=-15.0, max_value=15.0),
        )
        context_table = normalize.fit_transform(context_table)
    else:
        normalize = None

    return context_table.numerical.to(dtype), (prepare, normalize)


def transform_relative_time_features(
    values: Tensor,
    columns: tuple[str, ...],
    dtype: torch.dtype,
    processors: RelativeTimeProcessors,
) -> Tensor:
    """Transform relative-time features with context-fitted processors."""
    prepare, normalize = processors
    if prepare is None:
        return values.to(dtype)

    table = prepare.transform(_relative_time_table(values, columns=columns))
    if normalize is not None:
        table = normalize.transform(table)
    return table.numerical.to(dtype)


def _relative_time_table(
    values: Tensor,
    columns: tuple[str, ...],
) -> TableTensor:
    return TableTensor(
        columns={
            Stype.numerical: tuple(
                f"{column}__relative_time" for column in columns
            )
        },
        numerical=values,
    )
