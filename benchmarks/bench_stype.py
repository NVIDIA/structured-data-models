r"""Benchmark text stype inference across the PyArrow and cuDF backends.

Usage:

    python benchmarks/bench_stype.py

Runs every backend in turn, skipping cuDF when it is not installed.

Each cell reports the median and 90th percentile of ``REPEATS`` calls, and
the host-side peak allocation of a separate untimed call.

Axes:

* ``mode``: ``text`` enables text inference, ``baseline`` leaves it off, so
  the difference is the cost this inference path adds.
* ``profile``: the column shape, since the heuristic keys off unique ratio
  and word count rather than row count alone.
* ``rows``: straddles the 10,000 row sampling threshold in ``infer_stypes``;
  below it the full column is inspected, above it only the sample is.
* ``cols``: scales the per-column device synchronizations on cuDF.
"""

import importlib.util
import time
import tracemalloc
from collections.abc import Callable
from typing import Any

import pyarrow as pa
import torch
from sdm import Stype, infer_stypes

BACKENDS = ["arrow", "cudf"]
ROW_COUNTS = [1_000, 100_000]
COLUMN_COUNTS = [1, 10]
REPEATS = 5

_SENTENCE = "the product broke after one week of use"


def _prose(num_rows: int, column: int) -> list[Any]:
    # Long and unique: passes both halves of the heuristic.
    return [
        f"sentence number {row} of column {column} in this table"
        for row in range(num_rows)
    ]


def _short(num_rows: int, column: int) -> list[Any]:
    # Unique enough, but too few words to read as text.
    return [f"value {row % 50}" for row in range(num_rows)]


def _repeated(num_rows: int, column: int) -> list[Any]:
    # Long enough, but too few distinct values to read as text.
    return [_SENTENCE] * num_rows


def _mixed(num_rows: int, column: int) -> list[Any]:
    # Alternating prose and integers, as a realistic table shape.
    if column % 2 == 0:
        return _prose(num_rows=num_rows, column=column)
    return list(range(num_rows))


PROFILES: dict[str, Callable[[int, int], list[Any]]] = {
    "prose": _prose,
    "short": _short,
    "repeated": _repeated,
    "mixed": _mixed,
}

MODES: dict[str, set[Stype] | None] = {
    "text": {Stype.text},
    "baseline": None,
}


def _sync(backend: str) -> None:
    # cuDF work is asynchronous; flush it before reading the wall clock.
    if backend == "cudf":
        torch.cuda.synchronize()


def _table(
    backend: str,
    profile: str,
    num_rows: int,
    num_columns: int,
) -> Any:
    values = PROFILES[profile]
    data = {
        f"column_{column}": values(num_rows, column)
        for column in range(num_columns)
    }

    if backend == "arrow":
        return pa.table(data)

    import cudf

    return cudf.DataFrame(data)


def _peak_mebibytes(table: Any, allowed_stypes: set[Stype] | None) -> float:
    # Host allocations only; device memory is invisible to tracemalloc.
    tracemalloc.start()
    infer_stypes(table, allowed_stypes=allowed_stypes, seed=0)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return peak / 1024**2


def bench(backend: str, repeats: int) -> None:
    for profile in PROFILES:
        for mode, allowed_stypes in MODES.items():
            for num_rows in ROW_COUNTS:
                for num_columns in COLUMN_COUNTS:
                    # Table construction is excluded: for cuDF it is a
                    # host-to-device transfer that dwarfs the inference.
                    table = _table(
                        backend=backend,
                        profile=profile,
                        num_rows=num_rows,
                        num_columns=num_columns,
                    )

                    # Warm up CUDA context creation and kernel loading:
                    infer_stypes(
                        table,
                        allowed_stypes=allowed_stypes,
                        seed=0,
                    )
                    _sync(backend)

                    times = []
                    for _ in range(repeats):
                        start = time.perf_counter()
                        infer_stypes(
                            table,
                            allowed_stypes=allowed_stypes,
                            seed=0,
                        )
                        _sync(backend)
                        times.append(time.perf_counter() - start)

                    times.sort()
                    median = times[len(times) // 2]
                    p90 = times[int(0.9 * (len(times) - 1))]
                    peak = _peak_mebibytes(
                        table=table,
                        allowed_stypes=allowed_stypes,
                    )

                    print(
                        f"{backend:5s} {profile:8s} {mode:8s} "
                        f"rows={num_rows:>7,} cols={num_columns:>3} "
                        f"median={median * 1e3:8.2f} ms "
                        f"p90={p90 * 1e3:8.2f} ms "
                        f"({median / num_columns * 1e3:6.2f} ms/col) "
                        f"peak={peak:7.2f} MiB"
                    )


def main() -> None:
    for backend in BACKENDS:
        if backend == "cudf" and importlib.util.find_spec("cudf") is None:
            print("cudf: not installed, skipping")
            continue

        bench(backend=backend, repeats=REPEATS)


if __name__ == "__main__":
    main()
