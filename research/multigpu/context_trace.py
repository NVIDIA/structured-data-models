# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Summarize per-process CUDA activity within context benchmark NVTX ranges."""

import argparse
import json
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any


def union_ns(intervals: list[tuple[int, int]]) -> int:
    """Return interval-union duration, not overlapping kernel-time sums."""
    end, total = 0, 0
    for start, stop in sorted(intervals):
        total += max(0, stop - max(start, end))
        end = max(end, stop)
    return total


def analyze(database: Path) -> dict[str, Any]:
    """Read one immutable Nsight SQLite export and retain scoped evidence."""
    connection = sqlite3.connect(
        f"file:{database.resolve()}?mode=ro", uri=True
    )
    strings = dict(connection.execute("SELECT id,value FROM StringIds"))
    report = {"database": str(database), "phases": []}
    ranges = defaultdict(list)
    for start, end, tid, phase in connection.execute(
        "SELECT start,end,globalTid,text FROM NVTX_EVENTS "
        "WHERE text IN ('context_fit','context_predict_batch') "
        "AND end IS NOT NULL ORDER BY start"
    ):
        # Nsight globalTid reserves its low 24 bits for the local thread ID.
        ranges[(tid & ~((1 << 24) - 1), phase)].append((start, end))
    for (pid, phase), intervals in ranges.items():
        names = defaultdict(lambda: [0, 0])
        copies = defaultdict(lambda: [0, 0, 0])
        apis = defaultdict(lambda: [0, 0])
        kernels, communication = [], []
        devices = set()
        for start, end in intervals:
            for a, b, device, name in connection.execute(
                "SELECT start,end,deviceId,demangledName "
                "FROM CUPTI_ACTIVITY_KIND_KERNEL "
                "WHERE globalPid=? AND start>=? AND end<=?",
                (pid, start, end),
            ):
                name = strings[name]
                names[name][0] += 1
                names[name][1] += b - a
                kernels.append((a, b))
                devices.add(device)
                if "nccl" in name.lower():
                    communication.append((a, b))
            for label, size, duration in connection.execute(
                "SELECT e.label,c.bytes,c.end-c.start "
                "FROM CUPTI_ACTIVITY_KIND_MEMCPY c "
                "JOIN ENUM_CUDA_MEMCPY_OPER e ON e.id=c.copyKind "
                "WHERE c.globalPid=? AND c.start>=? AND c.end<=?",
                (pid, start, end),
            ):
                copies[label][0] += 1
                copies[label][1] += size
                copies[label][2] += duration
            for name, duration in connection.execute(
                "SELECT nameId,end-start FROM CUPTI_ACTIVITY_KIND_RUNTIME "
                "WHERE (globalTid & ~16777215)=? AND start>=? AND end<=?",
                (pid, start, end),
            ):
                apis[strings[name]][0] += 1
                apis[strings[name]][1] += duration
        wall = sum(b - a for a, b in intervals)
        report["phases"].append(
            {
                "global_pid": pid,
                "devices": sorted(devices),
                "phase": phase,
                "ranges": len(intervals),
                "nvtx_wall_sum_ms": wall / 1e6,
                "nvtx_wall_span_ms": (intervals[-1][1] - intervals[0][0])
                / 1e6,
                "kernel_count": len(kernels),
                "kernel_sum_ms": sum(b - a for a, b in kernels) / 1e6,
                "kernel_union_ms": union_ns(kernels) / 1e6,
                "kernel_active_fraction": union_ns(kernels) / wall,
                "nccl_count": len(communication),
                "nccl_sum_ms": sum(b - a for a, b in communication) / 1e6,
                "nccl_union_ms": union_ns(communication) / 1e6,
                "kernels": {
                    name: {"count": count, "sum_ms": ns / 1e6}
                    for name, (count, ns) in sorted(
                        names.items(),
                        key=lambda item: item[1][1],
                        reverse=True,
                    )
                },
                "copies": {
                    name: {"count": count, "bytes": size, "sum_ms": ns / 1e6}
                    for name, (count, size, ns) in copies.items()
                },
                "runtime_apis": {
                    name: {"count": count, "summed_cpu_ms": ns / 1e6}
                    for name, (count, ns) in sorted(
                        apis.items(), key=lambda item: item[1][1], reverse=True
                    )
                },
            }
        )
    report["diagnostics"] = list(
        connection.execute("SELECT severity,text FROM DIAGNOSTIC_EVENT")
    )
    connection.close()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(analyze(args.database), indent=2) + "\n")
