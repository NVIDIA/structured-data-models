"""Bounded text construction preserves the public whole-frame values."""

import numpy as np
import pandas as pd
import pytest
import torch
from examples.kumo.relational._relarena.adapter import (
    TEXT_TABLE_CHUNK_ROWS,
    tensorize_table,
)

import sdm


@pytest.mark.parametrize("rows", [3, TEXT_TABLE_CHUNK_ROWS + 3])
def test_mixed_text_table_matches_whole_frame(rows: int) -> None:
    index = np.arange(rows)
    frame = pd.DataFrame(
        {
            "entity": index + 100,
            "foreign": pd.array(index % 7, dtype="Int64"),
            "time": pd.date_range("2020-01-01", periods=rows, freq="h"),
            "value": index.astype(float),
            "category": np.where(
                index < TEXT_TABLE_CHUNK_ROWS, "first", "later"
            ),
            "text_a": [f"Unicode café 東京 {i}" for i in index],
            "text_b": [f"second text {i}" for i in index],
        }
    )
    frame.loc[0, ["foreign", "time", "value", "category", "text_a"]] = None
    frame.loc[1, "text_b"] = ""
    frame.loc[rows - 1, "text_b"] = None
    stypes = {
        "entity": "id",
        "foreign": "id",
        "time": "datetime",
        "value": "numerical",
        "category": "categorical",
        "text_a": "text",
        "text_b": "text",
    }
    expected = sdm.TableTensor.from_pandas(frame, stypes=stypes)
    actual = tensorize_table(frame, stypes)
    assert actual.columns == expected.columns
    pd.testing.assert_frame_equal(actual.to_pandas(), expected.to_pandas())
    torch.testing.assert_close(
        actual.numerical, expected.numerical, equal_nan=True
    )
    # Exercise a later sampling/gather boundary too, including chunk edges.
    selected = torch.tensor(
        [rows - 1, 0, min(TEXT_TABLE_CHUNK_ROWS, rows - 1), 1]
    )
    pd.testing.assert_frame_equal(
        actual[selected].to_pandas(), expected[selected].to_pandas()
    )


def test_empty_and_no_text_tables_keep_public_behavior() -> None:
    for rows in (0, TEXT_TABLE_CHUNK_ROWS + 1):
        frame = pd.DataFrame({"value": np.arange(rows, dtype=float)})
        expected = sdm.TableTensor.from_pandas(
            frame, stypes={"value": "numerical"}
        )
        actual = tensorize_table(frame, {"value": "numerical"})
        assert actual.columns == expected.columns
        torch.testing.assert_close(actual.numerical, expected.numerical)
