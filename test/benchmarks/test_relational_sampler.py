import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pyarrow as pa
import pytest
from sdm import Stype

_MODULE_NAME = "_relational_sampler_benchmark"
_MODULE_PATH = Path(__file__).parents[2] / "benchmarks/relational_sampler.py"
_SPEC = importlib.util.spec_from_file_location(_MODULE_NAME, _MODULE_PATH)
assert _SPEC is not None
assert _SPEC.loader is not None
_BENCHMARK = importlib.util.module_from_spec(_SPEC)
sys.modules[_MODULE_NAME] = _BENCHMARK
_SPEC.loader.exec_module(_BENCHMARK)

_load_json = _BENCHMARK._load_json
_hash_arrow = _BENCHMARK._hash_arrow
_timeline_payload = _BENCHMARK._timeline_payload
compare_results = _BENCHMARK.compare_results
parse_fanout = _BENCHMARK.parse_fanout
parse_positive_ints = _BENCHMARK.parse_positive_ints
percentile = _BENCHMARK.percentile
relational_inputs_from_database = _BENCHMARK.relational_inputs_from_database
summarize = _BENCHMARK.summarize
workload_kind = _BENCHMARK.workload_kind
write_timeline_artifacts = _BENCHMARK.write_timeline_artifacts


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


def test_arrow_hash_ignores_environment_metadata() -> None:
    table = pa.table({"value": [1, 2, 3]})

    assert _hash_arrow(table) == _hash_arrow(
        table.replace_schema_metadata({b"producer": b"different"})
    )


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
        "warmups": 3,
        "repetitions": 10,
        "task_selection": {
            "count": 4,
            "position_prefix": [0, 1, 2, 3],
            "positions_sha256": "positions",
        },
        "latency_ms": {"median": 2.0},
        "sampled_rows": {"median": 8.0},
        "output_to_host_ms_excluded": {"median": 0.1},
        "validation_fingerprint_ms_excluded": {"median": 0.2},
        "invariants": {"canonical_output_sha256": "same"},
    }
    cpu = {
        "schema_version": 3,
        "mode": "cpu",
        "workload": workload,
        "random_state": 123,
        "environment": {"versions": {}},
        "initialization": {},
        "runs": [run],
    }
    cuda = {
        "schema_version": 3,
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
    assert result["comparisons"][0]["equal_sampled_row_count"]
    assert result["comparisons"][0]["cuda_sampled_row_throughput_ratio"] == 2.0


def test_result_comparison_rejects_measurement_count_mismatch() -> None:
    run = {
        "kind": "finite_stochastic",
        "fanout": [16],
        "batch_size": 4,
        "warmups": 3,
        "repetitions": 10,
        "task_selection": {"positions_sha256": "same"},
        "latency_ms": {"median": 2.0},
        "sampled_rows": {"median": 8.0},
        "invariants": {},
    }
    common = {
        "schema_version": 3,
        "workload": {"dataset": "rel-f1"},
        "random_state": 123,
        "environment": {"versions": {}},
        "initialization": {},
    }
    cpu = {**common, "mode": "cpu", "runs": [run]}
    cuda = {
        **common,
        "mode": "cuda",
        "runs": [{**run, "repetitions": 9}],
    }

    with pytest.raises(ValueError, match="measurement counts"):
        compare_results(cpu, cuda)


def test_result_comparison_rejects_runtime_version_mismatch() -> None:
    common = {
        "schema_version": 3,
        "workload": {"dataset": "rel-f1"},
        "random_state": 123,
        "initialization": {},
        "runs": [],
    }
    cpu = {
        **common,
        "mode": "cpu",
        "environment": {"versions": {"torch": "2.11.0+cu130"}},
    }
    cuda = {
        **common,
        "mode": "cuda",
        "environment": {"versions": {"torch": "nightly"}},
    }

    with pytest.raises(ValueError, match="runtime versions differ"):
        compare_results(cpu, cuda)


def test_phase_profile_uses_data_flow_order_and_discloses_aggregation(
    tmp_path: Path,
) -> None:
    result = {
        "mode": "cuda",
        "runs": [
            {
                "fanout": [16],
                "batch_size": 8,
                "latency_ms": {"median": 4.0},
                "profiled_latency_ms": {"median": 4.5},
                "phases_ms": {
                    "synchronization_other_ms": {"median": 0.5},
                    "output_assembly_ms": {"median": 1.0},
                    "neighbor_sampling_ms": {"median": 2.0},
                    "task_to_seed_join_ms": {"median": 0.75},
                },
            }
        ],
    }

    payload = _timeline_payload(result)
    assert list(payload["runs"][0]["phases_ms"]) == [
        "task_to_seed_join_ms",
        "neighbor_sampling_ms",
        "output_assembly_ms",
        "synchronization_other_ms",
    ]

    html_path = tmp_path / "profile.html"
    trace_path = tmp_path / "profile.json"
    write_timeline_artifacts(result, html_path, trace_path)
    trace = json.loads(trace_path.read_text())

    assert [event["name"] for event in trace["traceEvents"]] == list(
        payload["runs"][0]["phases_ms"]
    )
    assert "not one request" in trace["metadata"]["source"]
    assert "not a single-request" in html_path.read_text()
