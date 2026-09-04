# ruff: noqa: D101

from typing import Any, cast

import torch
from torch.nn import GELU, LayerNorm, Linear, Sequential

from sdm.nn import QASSMax, RotaryEmbedding, TransformerBlock


class TabICLv2TransformerBlock(TransformerBlock):
    def __init__(
        self,
        channels: int,
        num_heads: int,
        norm_bias: bool,
        qassmax: bool,
        rope: RotaryEmbedding | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}

        norm = LayerNorm(channels, bias=norm_bias, **factory_kwargs)
        mlp = Sequential(
            LayerNorm(channels, bias=norm_bias, **factory_kwargs),
            Linear(channels, 2 * channels, **factory_kwargs),
            GELU(),
            Linear(2 * channels, channels, **factory_kwargs),
        )
        torch.nn.init.zeros_(cast(Linear, mlp[-1]).weight)
        torch.nn.init.zeros_(cast(Linear, mlp[-1]).bias)

        super().__init__(
            channels=channels,
            num_query_heads=num_heads,
            mlp=mlp,
            query_norm=norm,
            key_value_norm=norm,
            query_scaling=QASSMax(
                channels=channels // num_heads,
                num_heads=num_heads,
                **factory_kwargs,
            )
            if qassmax
            else None,
            query_transform=rope,
            key_transform=rope,
            **factory_kwargs,
        )

    def peak_bytes_per_example(
        self,
        query_length: int,
        key_value_length: int | None = None,
        *,
        dtype: torch.dtype,
    ) -> int:
        key_value_length = (
            query_length if key_value_length is None else key_value_length
        )
        length = max(query_length, key_value_length)
        precision = torch.empty((), dtype=dtype).element_size()
        factor = 15 if precision <= 2 else 22
        return factor * length * self.attn.q_dim


if __name__ == "__main__":
    import gc
    import itertools

    from tqdm import tqdm

    @torch.inference_mode
    def measure_peak(
        *,
        batch_size: int,
        length: int,
        channels: int,
        num_heads: int,
        amp: bool,
        warmups: int = 3,
        repeats: int = 5,
    ) -> int:
        block = TabICLv2TransformerBlock(
            channels=channels,
            num_heads=num_heads,
            norm_bias=True,
            qassmax=True,
            device=torch.device("cuda"),
        ).eval()

        x = torch.randn(
            (batch_size, length, channels),
            dtype=torch.float16 if amp else torch.float32,
            device=torch.device("cuda"),
        )

        peaks = []
        for _ in range(warmups + repeats):
            with torch.autocast("cuda", torch.float16, enabled=amp):
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                baseline = torch.cuda.memory_allocated()
                out = block(x, x)
                torch.cuda.synchronize()
                peaks.append(torch.cuda.max_memory_allocated() - baseline)
                del out

        return sum(peaks[warmups:]) / repeats

    configs = itertools.product(
        [128, 256, 512],
        [(128, 8), (512, 8)],
        [64, 128, 256, 512, 1024],
        [False, True],
    )
    measurements: list[tuple[int, int, int, bool, float]] = []
    for batch_size, (channels, heads), length, amp in tqdm(list(configs)):
        peak = measure_peak(
            batch_size=batch_size,
            length=length,
            channels=channels,
            num_heads=heads,
            amp=amp,
        )
        measurements.append((batch_size, channels, length, amp, peak))

    result = {}
    for amp in [False, True]:
        rows = [row for row in measurements if row[4] == amp]
        xs = [
            batch_size * length * channels
            for (batch_size, channels, length, _, _) in rows
        ]
        ys = [peak for *_, peak in rows]
        slope = sum(x * y for x, y in zip(xs, ys)) / sum(x * x for x in xs)
        preds = [slope * x for x in xs]
        mean = sum(ys) / len(ys)
        sse = sum((y - pred) ** 2 for y, pred in zip(ys, preds))
        sst = sum((y - mean) ** 2 for y in ys)
        result["fp16_amp" if amp else "fp32"] = (slope, 1.0 - sse / sst)

    for dtype, (slope, r2) in result.items():
        print(
            f"{dtype}: peak ~= batch_size * {slope:.6f} * length * channels; "
            f"R2={r2:.6f}"
        )
