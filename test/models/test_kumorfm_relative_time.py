import torch
from sdm import ColumnarTensor, RelatedTables, Stype, TableTensor
from sdm.models.kumorfm.graph import HomogeneousGraph
from sdm.models.kumorfm.relative_time import (
    fit_relative_time_features,
    propagate_anchor_indices,
    relative_days,
    relative_days_for_rows,
    transform_relative_time_features,
)
from sdm.processing.datetime import US_PER_DAY, US_PER_HOUR
from sdm.testing import withCUDA


@withCUDA
def test_relative_days_preserves_precision_and_missing(
    device: torch.device,
) -> None:
    missing = torch.iinfo(torch.int64).min
    offset = 100_000 * US_PER_DAY
    anchor = torch.tensor(
        [offset + 6 * US_PER_HOUR, missing],
        device=device,
    )
    timestamps = torch.tensor(
        [
            [offset - 2 * US_PER_DAY + 18 * US_PER_HOUR, missing],
            [offset, offset],
        ],
        device=device,
    )

    out = relative_days(anchor=anchor, timestamps=timestamps)

    assert out.dtype == torch.float32
    assert out.device == device
    torch.testing.assert_close(
        out,
        torch.tensor(
            [[1.5, float("nan")], [float("nan"), float("nan")]],
            device=device,
        ),
        equal_nan=True,
    )


@withCUDA
def test_relative_time_propagates_anchors_and_fits_on_context(
    device: torch.device,
) -> None:
    context_graph, context_timestamps = _graph_and_timestamps(
        group_ids=[1, 2],
        timestamps=[8, 15],
        device=device,
    )
    query_graph, query_timestamps = _graph_and_timestamps(
        group_ids=[3, 4],
        timestamps=[27, 35],
        device=device,
    )
    context_anchor = torch.tensor([10, 20], device=device) * US_PER_DAY
    query_anchor = torch.tensor([30, 40], device=device) * US_PER_DAY

    context_anchor_index = propagate_anchor_indices(
        num_anchors=2,
        root_index=torch.tensor([1, 0], device=device),
        graph=context_graph,
        num_hops=1,
    )[2:]
    query_anchor_index = propagate_anchor_indices(
        num_anchors=2,
        root_index=torch.tensor([1, 0], device=device),
        graph=query_graph,
        num_hops=1,
    )[2:]
    context, processors = fit_relative_time_features(
        context=relative_days_for_rows(
            anchor=context_anchor,
            anchor_index=context_anchor_index,
            timestamps=context_timestamps,
        ),
        columns=("created_at",),
        dtype=torch.float16,
    )
    query = transform_relative_time_features(
        values=relative_days_for_rows(
            anchor=query_anchor,
            anchor_index=query_anchor_index,
            timestamps=query_timestamps,
        ),
        columns=("created_at",),
        dtype=torch.float16,
        processors=processors,
    )

    assert context.dtype == query.dtype == torch.float16
    torch.testing.assert_close(
        context.float(),
        torch.tensor([[-1.0], [1.0]], device=device),
        atol=1e-3,
        rtol=1e-3,
    )
    torch.testing.assert_close(
        query.float(),
        torch.tensor([[-1.0 / 3.0], [1.0]], device=device),
        atol=1e-3,
        rtol=1e-3,
    )


def _graph_and_timestamps(
    group_ids: list[int],
    timestamps: list[int],
    device: torch.device,
) -> tuple[HomogeneousGraph, torch.Tensor]:
    users = TableTensor(
        columns={Stype.id: ("group_id",)},
        id=ColumnarTensor((torch.tensor(group_ids[::-1], device=device),)),
    )
    timestamp = torch.tensor(timestamps, device=device).unsqueeze(-1)
    groups = TableTensor(
        columns={
            Stype.datetime: ("created_at",),
            Stype.id: ("group_id",),
        },
        datetime=timestamp * US_PER_DAY,
        id=ColumnarTensor((torch.tensor(group_ids, device=device),)),
    )
    related_tables = RelatedTables(
        tables={"users": users, "groups": groups},
        relationships=[
            {
                "left_table": "users",
                "left_column": "group_id",
                "right_table": "groups",
                "right_column": "group_id",
            }
        ],
        task_links=[],
    )
    return (
        HomogeneousGraph.from_tables(
            tables=related_tables.tables,
            relationships=related_tables.relationships,
        ),
        groups.datetime,
    )
