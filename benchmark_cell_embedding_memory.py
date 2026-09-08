# ruff: noqa: T201

import argparse
import gc
import itertools
import time
from collections.abc import Sequence

import torch

from sdm.models.tabfm.cell_embedding import CellEmbedding


def _parse_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",")]


def _fit_no_intercept(
    xs: Sequence[int], ys: Sequence[int]
) -> tuple[float, float]:
    slope = sum(x * y for x, y in zip(xs, ys)) / sum(x * x for x in xs)
    preds = [slope * x for x in xs]
    mean = sum(ys) / len(ys)
    sse = sum((y - pred) ** 2 for y, pred in zip(ys, preds))
    sst = sum((y - mean) ** 2 for y in ys)
    return slope, 1.0 - sse / sst


@torch.inference_mode()
def _measure_cell_embedding(
    *,
    batch_size: int,
    rows: int,
    cols: int,
    channels: int,
    group_size: int,
    num_frequencies: int,
    amp: bool,
    device: torch.device,
    warmups: int,
    repeats: int,
) -> tuple[float, int]:
    module = CellEmbedding(
        channels=channels,
        group_size=group_size,
        num_frequencies=num_frequencies,
        device=device,
    ).eval()
    x = torch.randn(batch_size, rows, cols, device=device)
    categorical_mask = torch.randint(
        0,
        2,
        (batch_size, cols),
        device=device,
        dtype=torch.bool,
    )

    times: list[float] = []
    peaks: list[int] = []
    for i in range(warmups + repeats):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        baseline = torch.cuda.memory_allocated(device)

        torch.cuda.synchronize(device)
        start = time.perf_counter()
        with torch.autocast(device.type, torch.float16, enabled=amp):
            out = module(
                x=x,
                categorical_mask=categorical_mask,
                # batch_size_limit="auto",
            )
        torch.cuda.synchronize(device)

        if i >= warmups:
            times.append(time.perf_counter() - start)
            peaks.append(torch.cuda.max_memory_allocated(device) - baseline)
        del out

    return sum(times) / len(times), max(peaks)


def main() -> None:
    r"""Benchmark unchunked TabFM/Kumo cell embedding peak memory."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-sizes", default="1,2,4")
    parser.add_argument("--rows", default="4096,8096,16000")
    parser.add_argument("--cols", default="32,64")
    parser.add_argument("--channels", default="256")
    parser.add_argument("--group-size", type=int, default=3)
    parser.add_argument("--num-frequencies", type=int, default=32)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")

    device = torch.device("cuda")
    batch_sizes = _parse_ints(args.batch_sizes)
    rows = _parse_ints(args.rows)
    cols = _parse_ints(args.cols)
    channels = _parse_ints(args.channels)

    measurements: list[tuple[int, int, int, int, bool, float, int]] = []
    print(
        "batch_size,rows,cols,channels,group_size,num_frequencies,"
        "dtype,amp,seconds,measured_peak_bytes,measured_peak_mib"
    )

    configs = itertools.product(
        batch_sizes,
        rows,
        cols,
        channels,
        [False, True],
    )
    for batch_size, num_rows, num_cols, num_channels, amp in configs:
        seconds, peak = _measure_cell_embedding(
            batch_size=batch_size,
            rows=num_rows,
            cols=num_cols,
            channels=num_channels,
            group_size=args.group_size,
            num_frequencies=args.num_frequencies,
            amp=amp,
            device=device,
            warmups=args.warmups,
            repeats=args.repeats,
        )
        measurements.append(
            (batch_size, num_rows, num_cols, num_channels, amp, seconds, peak)
        )
        print(
            f"{batch_size},"
            f"{num_rows},"
            f"{num_cols},"
            f"{num_channels},"
            f"{args.group_size},"
            f"{args.num_frequencies},"
            f"{'fp16' if amp else 'fp32'},"
            f"{amp},"
            f"{seconds:.6f},"
            f"{peak},"
            f"{peak / 1024**2:.6f}",
            flush=True,
        )

    print()
    for amp in [False, True]:
        rows_for_mode = [row for row in measurements if row[4] == amp]
        element_size = 2 if amp else 4
        xs = [
            batch_size * num_rows * num_cols * num_channels * element_size
            for batch_size, num_rows, num_cols, num_channels, _, _, _ in rows_for_mode
        ]
        ys = [peak for *_, peak in rows_for_mode]
        slope, r2 = _fit_no_intercept(xs, ys)
        mode = "mixed" if amp else "single"
        print(
            f"{mode:>6} ~= batch_size * rows * cols * channels "
            f"* element_size * {slope:7.4f}; R2={r2:.4f}"
        )


if __name__ == "__main__":
    main()
