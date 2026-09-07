import argparse
import gc
import time

import numpy as np
import torch


def _parse_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",")]


def _parse_options(value: str) -> list[str]:
    return [item.strip() for item in value.split(",")]


def _make_data(
    context_rows: int,
    query_rows: int,
    cols: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_context = np.random.randn(context_rows, cols).astype(np.float32)
    y_context = np.random.randn(context_rows).astype(np.float32)
    x_query = np.random.randn(query_rows, cols).astype(np.float32)
    return x_context, y_context, x_query


def _make_regressor(
    *,
    n_estimators: int,
    batch_size: int,
    use_amp: str,
    use_fa3: str,
    offload_mode: str,
    kv_cache: bool,
    allow_auto_download: bool,
    verbose: bool,
) -> object:
    try:
        from tabicl import TabICLRegressor
    except ImportError as e:
        raise RuntimeError(
            "The official TabICL package is not installed. Install it with "
            "`pip install tabicl` in the environment used for this benchmark."
        ) from e

    return TabICLRegressor(
        n_estimators=n_estimators,
        batch_size=batch_size,
        kv_cache=kv_cache,
        allow_auto_download=allow_auto_download,
        device="cuda",
        use_amp=use_amp,
        use_fa3=use_fa3,
        offload_mode=offload_mode,
        random_state=42,
        verbose=verbose,
    )


def _measure_fit_predict(
    *,
    x_context: np.ndarray,
    y_context: np.ndarray,
    x_query: np.ndarray,
    n_estimators: int,
    batch_size: int,
    use_amp: str,
    use_fa3: str,
    offload_mode: str,
    kv_cache: bool,
    allow_auto_download: bool,
    verbose: bool,
    repeats: int,
) -> tuple[float, int]:
    device = torch.device("cuda")
    times = []
    peaks = []

    for _ in range(repeats):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        baseline = torch.cuda.memory_allocated(device)
        torch.cuda.synchronize(device)

        estimator = _make_regressor(
            n_estimators=n_estimators,
            batch_size=batch_size,
            use_amp=use_amp,
            use_fa3=use_fa3,
            offload_mode=offload_mode,
            kv_cache=kv_cache,
            allow_auto_download=allow_auto_download,
            verbose=verbose,
        )

        start = time.perf_counter()
        estimator.fit(x_context, y_context)
        pred = estimator.predict(x_query)
        torch.cuda.synchronize(device)
        times.append(time.perf_counter() - start)
        peaks.append(torch.cuda.max_memory_allocated(device) - baseline)

        del pred, estimator

    return sum(times) / len(times), max(peaks)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--context-rows", default="10000,50000")
    parser.add_argument("--query-rows", default="1000")
    parser.add_argument("--cols", default="100,200")
    parser.add_argument("--n-estimators", type=int, default=8)
    parser.add_argument("--batch-sizes", default="1,2,4,8")
    parser.add_argument("--use-amp", default="auto")
    parser.add_argument("--use-fa3", default="auto")
    parser.add_argument("--offload-modes", default="auto")
    parser.add_argument("--kv-cache", action="store_true")
    parser.add_argument("--allow-auto-download", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")

    print(
        "context_rows,query_rows,cols,n_estimators,batch_size,use_amp,"
        "use_fa3,offload_mode,kv_cache,seconds,peak_bytes,peak_mib"
    )

    for context_rows in _parse_ints(args.context_rows):
        for query_rows in _parse_ints(args.query_rows):
            for cols in _parse_ints(args.cols):
                x_context, y_context, x_query = _make_data(
                    context_rows=context_rows,
                    query_rows=query_rows,
                    cols=cols,
                )
                for batch_size in _parse_ints(args.batch_sizes):
                    for offload_mode in _parse_options(args.offload_modes):
                        seconds, peak = _measure_fit_predict(
                            x_context=x_context,
                            y_context=y_context,
                            x_query=x_query,
                            n_estimators=args.n_estimators,
                            batch_size=batch_size,
                            use_amp=args.use_amp,
                            use_fa3=args.use_fa3,
                            offload_mode=offload_mode,
                            kv_cache=args.kv_cache,
                            allow_auto_download=args.allow_auto_download,
                            verbose=args.verbose,
                            repeats=args.repeats,
                        )
                        print(
                            f"{context_rows},"
                            f"{query_rows},"
                            f"{cols},"
                            f"{args.n_estimators},"
                            f"{batch_size},"
                            f"{args.use_amp},"
                            f"{args.use_fa3},"
                            f"{offload_mode},"
                            f"{args.kv_cache},"
                            f"{seconds:.6f},"
                            f"{peak},"
                            f"{peak / 1024**2:.6f}",
                            flush=True,
                        )
                del x_context, y_context, x_query


if __name__ == "__main__":
    main()
