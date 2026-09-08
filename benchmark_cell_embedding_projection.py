# ruff: noqa: T201

import argparse
import gc
import itertools
import time
from collections.abc import Callable

import torch
from torch import Tensor
from torch.nn import functional as F


def _parse_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",")]


def _current_projection(
    fourier: Tensor,  # [B, R, C, G, 2F]
    mask: Tensor,  # [B, C, G]
    num_weight: Tensor,  # [D, 2F]
    num_bias: Tensor,  # [D]
    cat_weight: Tensor,  # [D, 2F]
    cat_bias: Tensor,  # [D]
) -> Tensor:  # [B, R, C, D]
    mask = mask.unsqueeze(1).unsqueeze(-1)  # [B, 1, C, G, 1]
    x = torch.where(
        mask,
        F.linear(fourier, cat_weight, cat_bias),
        F.linear(fourier, num_weight, num_bias),
    )
    return x.sum(dim=-2)


def _selected_weight_projection(
    fourier: Tensor,  # [B, R, C, G, 2F]
    mask: Tensor,  # [B, C, G]
    num_weight: Tensor,  # [D, 2F]
    num_bias: Tensor,  # [D]
    cat_weight: Tensor,  # [D, 2F]
    cat_bias: Tensor,  # [D]
) -> Tensor:  # [B, R, C, D]
    weight = torch.where(
        mask[..., None, None],  # [B, C, G, 1, 1]
        cat_weight.view(1, 1, 1, *cat_weight.size()),
        num_weight.view(1, 1, 1, *num_weight.size()),
    )  # [B, C, G, D, 2F]
    bias = torch.where(
        mask[..., None],  # [B, C, G, 1]
        cat_bias.view(1, 1, 1, -1),
        num_bias.view(1, 1, 1, -1),
    )  # [B, C, G, D]
    x = torch.einsum("brcgf,bcgdf->brcd", fourier, weight)
    return x + bias.sum(dim=-2).unsqueeze(1)


@torch.inference_mode()
def _measure(
    fn: Callable[[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor], Tensor],
    *,
    fourier: Tensor,
    mask: Tensor,
    num_weight: Tensor,
    num_bias: Tensor,
    cat_weight: Tensor,
    cat_bias: Tensor,
    amp: bool,
    warmups: int,
    repeats: int,
) -> tuple[float, int]:
    device = fourier.device
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
            out = fn(
                fourier,
                mask,
                num_weight,
                num_bias,
                cat_weight,
                cat_bias,
            )
        torch.cuda.synchronize(device)

        if i >= warmups:
            times.append(time.perf_counter() - start)
            peaks.append(torch.cuda.max_memory_allocated(device) - baseline)
        del out

    return sum(times) / len(times), max(peaks)


def main() -> None:
    r"""Benchmark cell embedding projection alternatives."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-sizes", default="1,2,4,8")
    parser.add_argument("--rows", default="1024,4096,8192")
    parser.add_argument("--cols", default="64,100,128")
    parser.add_argument("--channels", default="128,256")
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
    group_size = args.group_size
    num_features = 2 * args.num_frequencies

    print(
        "mode,batch_size,rows,cols,group_size,num_frequencies,channels,"
        "dtype,amp,seconds,peak_bytes,peak_mib,max_abs_diff"
    )

    configs = itertools.product(
        batch_sizes,
        rows,
        cols,
        channels,
        [False, True],
    )
    for batch_size, num_rows, num_cols, num_channels, amp in configs:
        fourier = torch.randn(
            batch_size,
            num_rows,
            num_cols,
            group_size,
            num_features,
            device=device,
        )
        mask = torch.randint(
            0,
            2,
            (batch_size, num_cols, group_size),
            device=device,
            dtype=torch.bool,
        )
        num_weight = torch.randn(num_channels, num_features, device=device)
        cat_weight = torch.randn(num_channels, num_features, device=device)
        num_bias = torch.randn(num_channels, device=device)
        cat_bias = torch.randn(num_channels, device=device)

        with torch.autocast(device.type, torch.float16, enabled=amp):
            expected = _current_projection(
                fourier,
                mask,
                num_weight,
                num_bias,
                cat_weight,
                cat_bias,
            )
            actual = _selected_weight_projection(
                fourier,
                mask,
                num_weight,
                num_bias,
                cat_weight,
                cat_bias,
            )
        torch.cuda.synchronize(device)
        max_abs_diff = (expected - actual).abs().max().item()
        del expected, actual

        for name, fn in [
            ("where", _current_projection),
            ("selected_einsum", _selected_weight_projection),
        ]:
            seconds, peak = _measure(
                fn,
                fourier=fourier,
                mask=mask,
                num_weight=num_weight,
                num_bias=num_bias,
                cat_weight=cat_weight,
                cat_bias=cat_bias,
                amp=amp,
                warmups=args.warmups,
                repeats=args.repeats,
            )
            print(
                f"{name},"
                f"{batch_size},"
                f"{num_rows},"
                f"{num_cols},"
                f"{group_size},"
                f"{args.num_frequencies},"
                f"{num_channels},"
                f"{'mixed' if amp else 'single'},"
                f"{amp},"
                f"{seconds:.6f},"
                f"{peak},"
                f"{peak / 1024**2:.6f},"
                f"{max_abs_diff:.6g}",
                flush=True,
            )

        del fourier, mask, num_weight, num_bias, cat_weight, cat_bias


if __name__ == "__main__":
    main()
