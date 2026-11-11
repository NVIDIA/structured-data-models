# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare raw per-processor results without imputing missing measurements."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


def _peak(data: dict[str, Any], field: str) -> int | None:
    samples = [
        entry[field]
        for entries in data.get("memory", {}).values()
        for entry in entries
    ]
    return max(samples) if samples else None


def _csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Compare total/lifecycle timings, allocations, and ATen operations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    args = parser.parse_args()
    root = args.artifacts
    totals, processors, operations = [], [], []
    for path in sorted((root / "baseline").glob("*-cuda.json")):
        before = _read(path)
        after = _read(root / "after" / path.name)
        counters = _read(root / "counters" / path.name)
        if counters:
            assert counters["dataset_sha256"] == before["dataset_sha256"]
        if after:
            assert after["dataset_sha256"] == before["dataset_sha256"]
        total = {"case": before["case"]}
        for label, data in (("before", before), ("after", after)):
            for lifecycle in (
                "fit_transform",
                "transform",
                "total",
                "features_fit_unfused",
            ):
                timing = data.get("latency", {}).get(lifecycle, {})
                for statistic in ("median_ms", "p95_ms"):
                    total[f"{label}_{lifecycle}_{statistic}"] = timing.get(
                        statistic
                    )
            total[f"{label}_cpu_peak_rss_bytes"] = _peak(
                data, "cpu_peak_rss_bytes"
            )
            total[f"{label}_gpu_peak_allocated_bytes"] = _peak(
                data, "gpu_peak_allocated_bytes"
            )
        total["speedup_percent"] = (
            (
                1
                - total["after_total_median_ms"]
                / total["before_total_median_ms"]
            )
            * 100
            if after
            else None
        )
        for device, metric in (
            ("cpu", "cpu_peak_rss_bytes"),
            ("gpu", "gpu_peak_allocated_bytes"),
        ):
            first, last = total[f"before_{metric}"], total[f"after_{metric}"]
            total[f"{device}_peak_memory_delta_percent"] = (
                (last / first - 1) * 100
                if first and last is not None
                else None
            )
        totals.append(total)
        timings = [
            {
                (r["processor"], r["operation"]): r
                for r in data.get("processors", [])
            }
            for data in (before, after)
        ]
        counts = [
            {
                (r["processor"], r["operation"]): r
                for r in data.get("processor_counters", [])
            }
            for data in (before, after or counters)
        ]
        memory_sources = [
            _read(root / "memory" / phase / path.name)
            for phase in ("baseline", "after")
        ]
        for measured, reference in zip(memory_sources, (before, after)):
            if measured and reference:
                assert measured["source_sha256"] == reference["source_sha256"]
                assert (
                    measured["dataset_sha256"] == reference["dataset_sha256"]
                )
        memory = [
            {
                (r["processor"], r["operation"]): r
                for r in measured.get("processor_memory", [])
            }
            for measured in memory_sources
        ]
        keys = set().union(*(mapping for mapping in (*timings, *counts)))
        for processor, lifecycle in sorted(keys):
            key = processor, lifecycle
            row = {
                "case": before["case"],
                "processor": processor,
                "lifecycle": lifecycle,
            }
            for label, data, timing, count, peaks in zip(
                ("before", "after"),
                (before, after),
                timings,
                counts,
                memory,
                strict=True,
            ):
                record, counter = timing.get(key, {}), count.get(key, {})
                for field in (
                    "cpu_peak_rss_bytes",
                    "cpu_rss_increase_bytes",
                    "gpu_peak_allocated_bytes",
                    "gpu_allocation_increase_bytes",
                ):
                    row[f"{label}_{field}"] = peaks.get(key, {}).get(field)
                row[f"{label}_callbacks_per_pipeline"] = (
                    record.get("calls", 0) / data["repeats"] if data else None
                )
                for device in ("cpu", "gpu"):
                    for statistic in ("median_ms", "p95_ms"):
                        row[f"{label}_{device}_{statistic}_per_callback"] = (
                            record.get(device, {}).get(statistic)
                        )
                row[f"{label}_exclusive_cpu_ms_per_pipeline"] = (
                    record.get("exclusive_cpu_ms", 0) / data["repeats"]
                    if record
                    else None
                )
                row[f"{label}_aten_calls"] = (
                    sum(counter.get("ops", {}).values()) if counter else None
                )
                for device in ("cpu", "cuda"):
                    row[f"{label}_{device}_allocation_events"] = (
                        counter.get("dense_allocations", {}).get(device, 0)
                        if counter
                        else None
                    )
                    row[f"{label}_{device}_allocated_bytes"] = (
                        counter.get("dense_allocated_bytes", {}).get(device, 0)
                        if counter
                        else None
                    )
            for metric in (
                "cpu_median_ms_per_callback",
                "cpu_p95_ms_per_callback",
                "gpu_median_ms_per_callback",
                "gpu_p95_ms_per_callback",
                "exclusive_cpu_ms_per_pipeline",
                "cpu_peak_rss_bytes",
                "gpu_peak_allocated_bytes",
            ):
                first, last = row[f"before_{metric}"], row[f"after_{metric}"]
                row[f"{metric}_delta_percent"] = (
                    (last / first - 1) * 100
                    if first and last is not None
                    else None
                )
            processors.append(row)
            for name in sorted(
                set(counts[0].get(key, {}).get("ops", {}))
                | set(counts[1].get(key, {}).get("ops", {}))
            ):
                operations.append(
                    {
                        "case": before["case"],
                        "processor": processor,
                        "lifecycle": lifecycle,
                        "operation": name,
                        "before_calls": counts[0]
                        .get(key, {})
                        .get("ops", {})
                        .get(name, 0),
                        "after_calls": counts[1]
                        .get(key, {})
                        .get("ops", {})
                        .get(name, 0)
                        if counters or after
                        else None,
                    }
                )
    _csv(root / "totals.csv", totals)
    _csv(root / "processors.csv", processors)
    _csv(root / "operations.csv", operations)
    lines = [
        "# Preprocessing measurement summary",
        "",
        "Empty after fields mean not measured under comparable conditions; "
        "no speedup is inferred.",
        "",
        "| Case | Before median / p95 ms | After median / p95 ms | "
        "Speedup | Before / after GPU peak GiB |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in totals:
        after = (
            f"{row['after_total_median_ms']:.2f} / "
            f"{row['after_total_p95_ms']:.2f}"
            if row["after_total_median_ms"] is not None
            else "pending"
        )
        speedup = (
            f"{row['speedup_percent']:+.1f}%"
            if row["speedup_percent"] is not None
            else "pending"
        )
        gpu_peak = (
            f"{row['after_gpu_peak_allocated_bytes'] / 2**30:.3f}"
            if row["after_gpu_peak_allocated_bytes"] is not None
            else "pending"
        )
        lines.append(
            f"| {row['case']} | {row['before_total_median_ms']:.2f} / "
            f"{row['before_total_p95_ms']:.2f} | {after} | {speedup} | "
            f"{row['before_gpu_peak_allocated_bytes'] / 2**30:.3f} / "
            f"{gpu_peak} |"
        )
    lines += [
        "",
        "## Physical ATen dispatch counts (including views)",
        "",
        "| Case | Before | After |",
        "| --- | ---: | ---: |",
    ]
    for row in totals:
        counts = Counter()
        for operation in operations:
            if operation["case"] == row["case"]:
                for phase in ("before", "after"):
                    if operation[f"{phase}_calls"] is not None:
                        counts[phase] += operation[f"{phase}_calls"]
        lines.append(
            f"| {row['case']} | {counts['before']} | "
            f"{counts.get('after', 'pending')} |"
        )
    lines += [
        "",
        "See processors.csv for every processor/configuration/lifecycle "
        "and operations.csv for copies, indexing, materialization and all "
        "other ATen operations. Allocation events count positive profiler "
        "self-memory events, not malloc calls. Inclusive callback times "
        "are not additive; callback counts can differ when batching changes.",
    ]
    (root / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print(root / "SUMMARY.md")


if __name__ == "__main__":
    main()
