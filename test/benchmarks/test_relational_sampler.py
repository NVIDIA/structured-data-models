import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from benchmarks.relational_sampler import (
    _load_json,
    compare_results,
    parse_fanout,
    parse_positive_ints,
    percentile,
    relational_inputs_from_database,
    summarize,
    workload_kind,
)
from sdm import Stype


def test_parse_request_shapes() -> None:
    assert parse_positive_ints("1,128,1024") == (1, 128, 1024)
    assert parse_fanout("16,16") == (16, 16)
    assert workload_kind((-1,)) == "exhaustive_parity"
    assert workload_kind((16, 16)) == "finite_stochastic"

    with pytest.raises(Exception, match="positive"):
        parse_positive_ints("0")
    with pytest.raises(Exception, match="-1"):
        parse_fanout("-2")


def test_timing_summary_has_interpolated_tail_and_spread() -> None:
    values = [1.0, 2.0, 3.0, 4.0, 5.0]

    assert percentile(values, 0.95) == pytest.approx(4.8)
    summary = summarize(values)

    assert summary["median"] == 3.0
    assert summary["p95"] == pytest.approx(4.8)
    assert summary["stdev"] > 0


def test_result_parser_requires_the_supported_schema(tmp_path: Path) -> None:
    result_path = tmp_path / "result.json"
    result_path.write_text(json.dumps({"schema_version": 0}))

    with pytest.raises(ValueError, match="Unsupported result schema"):
        _load_json(result_path)


def test_relbench_like_database_constructs_id_relationships() -> None:
    database = SimpleNamespace(
        table_dict={
            "users": SimpleNamespace(
                df=pd.DataFrame({"user": [1, 2], "joined": [1, 2]}),
                pkey_col="user",
                fkey_col_to_pkey_table={},
                time_col=None,
            ),
            "events": SimpleNamespace(
                df=pd.DataFrame(
                    {
                        "event": [10, 11],
                        "user": [1, 2],
                        "when": pd.to_datetime(["2020-01-01", "2020-01-02"]),
                    }
                ),
                pkey_col="event",
                fkey_col_to_pkey_table={"user": "users"},
                time_col="when",
            ),
        }
    )

    data, time_columns, identity_columns = relational_inputs_from_database(
        database
    )

    assert data.tables["users"].stype("user") == Stype.id
    assert data.tables["events"].stype("user") == Stype.id
    assert time_columns == {"events": "when"}
    assert identity_columns == {
        "users": ("user",),
        "events": ("event", "user"),
    }
    assert data.relationships[0].left_table == "events"
    assert data.relationships[0].right_table == "users"


def test_result_comparison_checks_workload_and_exhaustive_parity() -> None:
    workload = {"dataset": "rel-f1", "task": "driver-position"}
    run = {
        "kind": "exhaustive_parity",
        "fanout": [-1],
        "batch_size": 4,
        "task_selection": {
            "count": 4,
            "position_prefix": [0, 1, 2, 3],
            "positions_sha256": "positions",
        },
        "latency_ms": {"median": 2.0},
        "sampled_rows": {"median": 8.0},
        "invariants": {"canonical_output_sha256": "same"},
    }
    cpu = {
        "schema_version": 1,
        "mode": "cpu",
        "workload": workload,
        "random_state": 123,
        "environment": {"versions": {}},
        "initialization": {},
        "runs": [run],
    }
    cuda = {
        "schema_version": 1,
        "mode": "cuda",
        "workload": workload,
        "random_state": 123,
        "environment": {"versions": {}},
        "initialization": {},
        "runs": [{**run, "latency_ms": {"median": 1.0}}],
    }

    result = compare_results(cpu, cuda)

    assert result["exhaustive_parity_passed"]
    assert result["comparisons"][0]["cuda_speedup"] == 2.0
