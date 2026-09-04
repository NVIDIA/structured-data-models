import argparse
import gc
import os
import time
from collections.abc import Sequence

import torch

import sdm


def _parse_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",")]


def _parse_floats(value: str) -> list[float]:
    return [float(item) for item in value.split(",")]


def _make_table(
    rows: int,
    cols: int,
    *,
    device: torch.device,
) -> sdm.TableTensor:
    return sdm.TableTensor.from_tensor(
        torch.randn(rows, cols, device=device),
    )


def _measure_forward(
    model: sdm.models.TabICLv2,
    x_context: sdm.TableTensor,
    y_context: sdm.TableTensor,
    x_query: sdm.TableTensor,
    *,
    num_estimators: int,
    amp: bool,
    warmups: int,
    repeats: int,
) -> tuple[float, int]:
    device = x_context.device

    with torch.inference_mode():
        for _ in range(warmups):
            with torch.autocast(device.type, torch.float16, enabled=amp):
                out = model(
                    x_context=x_context,
                    y_context=y_context,
                    x_query=x_query,
                    num_estimators=num_estimators,
                )
            del out
            torch.cuda.synchronize(device)

        times = []
        peaks = []
        for _ in range(repeats):
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            baseline = torch.cuda.memory_allocated(device)

            torch.cuda.synchronize(device)
            start = time.perf_counter()
            with torch.autocast(device.type, torch.float16, enabled=amp):
                out = model(
                    x_context=x_context,
                    y_context=y_context,
                    x_query=x_query,
                    num_estimators=num_estimators,
                )
            torch.cuda.synchronize(device)
            times.append(time.perf_counter() - start)
            peaks.append(torch.cuda.max_memory_allocated(device) - baseline)
            del out

    return sum(times) / len(times), max(peaks)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--context-rows", default="128,256,512")
    parser.add_argument("--query-rows", default="64")
    parser.add_argument("--cols", default="16,32,64,128")
    parser.add_argument("--fractions", default="0.05,0.1,0.2,0.5")
    parser.add_argument("--num-estimators", type=int, default=1)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--pretrained", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")

    device = torch.device("cuda")
    model = sdm.models.TabICLv2(
        task=sdm.Task.regression,
        pretrained=args.pretrained,
        device=device,
    ).eval()

    old_fraction = os.environ.get("SDM_CHUNK_MEMORY_FRACTION")
    print(
        "memory_fraction,context_rows,query_rows,cols,num_estimators,"
        "amp,seconds,peak_bytes,peak_mib"
    )

    context_rows: Sequence[int] = _parse_ints(args.context_rows)
    query_rows: Sequence[int] = _parse_ints(args.query_rows)
    cols: Sequence[int] = _parse_ints(args.cols)
    fractions: Sequence[float] = _parse_floats(args.fractions)

    try:
        for memory_fraction in fractions:
            os.environ["SDM_CHUNK_MEMORY_FRACTION"] = str(memory_fraction)
            for num_context_rows in context_rows:
                for num_query_rows in query_rows:
                    for num_cols in cols:
                        x = _make_table(
                            num_context_rows + num_query_rows,
                            num_cols,
                            device=device,
                        )
                        y = _make_table(num_context_rows, 1, device=device)
                        x_context = x[:num_context_rows]
                        x_query = x[num_context_rows:]

                        seconds, peak = _measure_forward(
                            model=model,
                            x_context=x_context,
                            y_context=y,
                            x_query=x_query,
                            num_estimators=args.num_estimators,
                            amp=not args.no_amp,
                            warmups=args.warmups,
                            repeats=args.repeats,
                        )
                        print(
                            f"{memory_fraction},"
                            f"{num_context_rows},"
                            f"{num_query_rows},"
                            f"{num_cols},"
                            f"{args.num_estimators},"
                            f"{not args.no_amp},"
                            f"{seconds:.6f},"
                            f"{peak},"
                            f"{peak / 1024**2:.6f}",
                            flush=True,
                        )
                        del x, y, x_context, x_query
    finally:
        if old_fraction is None:
            os.environ.pop("SDM_CHUNK_MEMORY_FRACTION", None)
        else:
            os.environ["SDM_CHUNK_MEMORY_FRACTION"] = old_fraction


if __name__ == "__main__":
    main()
