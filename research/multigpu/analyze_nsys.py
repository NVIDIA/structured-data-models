# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Summarize CUDA activity inside named Nsight Systems NVTX ranges."""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any


def merged(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Union overlapping intervals before computing GPU activity."""
    result: list[tuple[int, int]] = []
    for start, end in sorted(intervals):
        if result and start <= result[-1][1]:
            result[-1] = result[-1][0], max(result[-1][1], end)
        else:
            result.append((start, end))
    return result


def analyze(path: Path, phase: str) -> dict[str, Any]:
    """Read a SQLite export without modifying it or the original trace."""
    connection = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    ranges = list(
        connection.execute(
            "SELECT start,end FROM NVTX_EVENTS "
            "WHERE text=? AND end IS NOT NULL",
            (phase,),
        )
    )
    strings = dict(connection.execute("SELECT id,value FROM StringIds"))
    report: dict[str, Any] = {
        "source": str(path),
        "phase": phase,
        "ranges": [],
    }
    for start, end in ranges:
        duration = end - start
        kernels = list(
            connection.execute(
                "SELECT start,end,deviceId,demangledName,correlationId "
                "FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE start>=? AND end<=?",
                (start, end),
            )
        )
        by_device: dict[int, list[tuple[int, int]]] = defaultdict(list)
        by_name: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        for a, b, device, name, _ in kernels:
            by_device[device].append((a, b))
            by_name[strings[name]][0] += 1
            by_name[strings[name]][1] += b - a
        devices = {}
        events: dict[int, int] = defaultdict(int)
        events[start] += 0
        events[end] += 0
        for device, intervals in by_device.items():
            union = merged(intervals)
            active = sum(b - a for a, b in union)
            devices[str(device)] = {
                "kernel_count": len(intervals),
                "kernel_sum_ms": sum(b - a for a, b in intervals) / 1e6,
                "kernel_union_ms": active / 1e6,
                "kernel_active_fraction": active / duration,
            }
            for a, b in union:
                events[a] += 1
                events[b] -= 1
        histogram: dict[int, int] = defaultdict(int)
        count, previous = 0, start
        for timestamp, delta in sorted(events.items()):
            histogram[count] += timestamp - previous
            count += delta
            previous = timestamp
        runtime = list(
            connection.execute(
                "SELECT s.value,COUNT(*),SUM(r.end-r.start),"
                "COUNT(DISTINCT r.globalTid) "
                "FROM CUPTI_ACTIVITY_KIND_RUNTIME r "
                "JOIN StringIds s ON s.id=r.nameId "
                "WHERE r.start>=? AND r.end<=? GROUP BY s.value "
                "ORDER BY SUM(r.end-r.start) DESC",
                (start, end),
            )
        )
        launches = list(
            connection.execute(
                "SELECT r.globalTid,COUNT(*),SUM(r.end-r.start) "
                "FROM CUPTI_ACTIVITY_KIND_RUNTIME r "
                "JOIN StringIds s ON s.id=r.nameId "
                "WHERE r.start>=? AND r.end<=? "
                "AND s.value LIKE '%LaunchKernel%' "
                "GROUP BY r.globalTid",
                (start, end),
            )
        )
        copies = list(
            connection.execute(
                "SELECT e.label,COUNT(*),SUM(c.bytes),SUM(c.end-c.start) "
                "FROM CUPTI_ACTIVITY_KIND_MEMCPY c "
                "JOIN ENUM_CUDA_MEMCPY_OPER e ON e.id=c.copyKind "
                "WHERE c.start>=? AND c.end<=? GROUP BY e.label",
                (start, end),
            )
        )
        report["ranges"].append(
            {
                "traced_wall_s": duration / 1e9,
                "devices": devices,
                "concurrent_gpu_kernel_fraction": {
                    str(n): t / duration for n, t in sorted(histogram.items())
                },
                "kernels": {
                    name: {"count": count, "sum_ms": ns / 1e6}
                    for name, (count, ns) in sorted(
                        by_name.items(),
                        key=lambda item: item[1][1],
                        reverse=True,
                    )
                },
                "runtime_apis": [
                    {
                        "name": name,
                        "calls": calls,
                        "summed_cpu_ms": ns / 1e6,
                        "threads": threads,
                    }
                    for name, calls, ns, threads in runtime
                ],
                "launch_threads": [
                    {
                        "global_tid": tid,
                        "launch_calls": calls,
                        "summed_cpu_ms": ns / 1e6,
                    }
                    for tid, calls, ns in launches
                ],
                "copies": [
                    {
                        "kind": kind,
                        "operations": calls,
                        "bytes": size,
                        "sum_ms": ns / 1e6,
                    }
                    for kind, calls, size, ns in copies
                ],
            }
        )
    report["diagnostics"] = list(
        connection.execute("SELECT severity,text FROM DIAGNOSTIC_EVENT")
    )
    connection.close()
    return report


def main() -> None:
    """Analyze one trace and write reproducible JSON evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", type=Path)
    parser.add_argument("--phase", default="prediction_pass")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = analyze(args.database, args.phase)
    payload = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(payload + "\n")
    else:
        print(payload)  # noqa: T201


if __name__ == "__main__":
    main()
