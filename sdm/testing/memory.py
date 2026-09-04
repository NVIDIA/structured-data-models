import gc
import itertools
from collections.abc import Callable, Sequence

import torch

from sdm.nn import TransformerBlock


def benchmark_transformer_block_memory_peak(
    block: Callable[[int, int], TransformerBlock],
    channels_and_heads: Sequence[tuple[int, int]],
    device: torch.device | str = "cuda",
) -> None:
    r"""Benchmark peak memory of a :class:`~sdm.nn.TransformerBlock`."""
    import tqdm

    @torch.inference_mode()
    def _run(
        batch_size: int,
        length: int,
        channels: int,
        num_heads: int,
        amp: bool,
        warmups: int = 3,
        repeats: int = 5,
    ) -> int:

        module = block(channels, num_heads).to(device).eval()
        x = torch.randn(
            (batch_size, length, channels),
            dtype=torch.float16 if amp else torch.float32,
            device=device,
        )

        peaks = []
        for _ in range(warmups + repeats):
            with torch.autocast(x.device.type, torch.float16, enabled=amp):
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                baseline = torch.cuda.memory_allocated()
                out = module(x, x)
                torch.cuda.synchronize()
                peaks.append(torch.cuda.max_memory_allocated() - baseline)
                del out

        return max(peaks[warmups:])

    configs = itertools.product(
        [128, 256, 512],  # batch_size
        [64, 128, 256, 512, 1024],  # length
        channels_and_heads,
        [False, True],  # amp
    )
    measurements: list[tuple[int, int, int, bool, float]] = []
    for batch_size, length, (channels, heads), amp in tqdm.tqdm(list(configs)):
        peak = _run(
            batch_size=batch_size,
            length=length,
            channels=channels,
            num_heads=heads,
            amp=amp,
        )
        measurements.append((batch_size, length, channels, amp, peak))

    result = {}
    for amp in [False, True]:
        rows = [row for row in measurements if row[3] == amp]
        xs = [
            batch_size * length * channels
            for (batch_size, length, channels, _, _) in rows
        ]
        ys = [peak for *_, peak in rows]
        slope = sum(x * y for x, y in zip(xs, ys)) / sum(x * x for x in xs)
        preds = [slope * x for x in xs]
        mean = sum(ys) / len(ys)
        sse = sum((y - pred) ** 2 for y, pred in zip(ys, preds))
        sst = sum((y - mean) ** 2 for y in ys)
        result["mixed" if amp else "single"] = (
            slope / (2 if amp else 4),
            1.0 - sse / sst,
        )

    for mode, (slope, r2) in result.items():
        print(
            f"{mode:>6} ~= batch_size * {slope:7.4f} * length * element_size "
            f"* channels; R2={r2:.4f}"
        )
