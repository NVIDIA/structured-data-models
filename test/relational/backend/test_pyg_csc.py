import pytest
import torch

from sdm.relational.backend._pyg_lib import _to_csc


@pytest.mark.parametrize("temporal", [False, True])
def test_csc_neighbors_are_independent_of_join_output_order(
    temporal: bool,
) -> None:
    source = torch.arange(60)
    edges = torch.stack((source, source % 3))
    times = source % 4 if temporal else None
    expected_row, expected_ptr = _to_csc(edges, 3, times)
    # Arrow's parallel hash join need not preserve source row ordering.
    for permutation in (source.flip(0), source.roll(17)):
        row, ptr = _to_csc(edges[:, permutation], 3, times)
        assert row.equal(expected_row)
        assert ptr.equal(expected_ptr)
        # Fixed-seed uniform positions and temporal-last positions must refer
        # to the same nodes across equivalent graph builds.
        for destination in range(3):
            end = int(ptr[destination + 1])
            assert row[end - 2 : end].equal(expected_row[end - 2 : end])


def test_temporal_csc_orders_ties_by_source_row() -> None:
    edges = torch.tensor([[5, 1, 3, 4, 0, 2], [0, 0, 0, 0, 0, 0]])
    times = torch.tensor([1, 0, 1, 0, 1, 0])
    row, ptr = _to_csc(edges, 1, times)
    assert row.tolist() == [1, 3, 5, 0, 2, 4]
    assert ptr.tolist() == [0, 6]
