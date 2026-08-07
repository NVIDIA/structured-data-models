import argparse
from pathlib import Path

import torch

from sdm import TableTensor
from sdm.models import TabICLv2


class _ProfileTabICLv2(TabICLv2):
    prefetch_caches = True

    def _should_prefetch_caches(self, x: TableTensor) -> bool:
        return self.prefetch_caches


def _profile(
    model: _ProfileTabICLv2,
    x_query: torch.Tensor,
    *,
    prefetch: bool,
    path: Path,
) -> torch.Tensor:
    model.prefetch_caches = prefetch
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        profile_memory=True,
        record_shapes=True,
    ) as profiler:
        out = model.predict(x_query)
        torch.cuda.synchronize()

    profiler.export_chrome_trace(str(path))
    mode = "async" if prefetch else "sync"
    print(f"\n{mode}: {path}")
    print(
        profiler.key_averages().table(
            sort_by="self_device_time_total",
            row_limit=15,
        )
    )
    return out


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("cache-prefetch-profiles"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for cache prefetch profiling")

    torch.manual_seed(0)
    device = torch.device("cuda:0")
    model = _ProfileTabICLv2(pretrained=False, device=device)
    x_context = torch.randn(128, 16, device=device)
    x_query = torch.randn(64, 16, device=device)
    y_context = torch.randn(128, 1, device=device)
    model.fit(x_context, y_context, num_estimators=3)

    for prefetch in (False, True):
        model.prefetch_caches = prefetch
        model.predict(x_query)
        torch.cuda.synchronize()

    sync = _profile(
        model,
        x_query,
        prefetch=False,
        path=args.output_dir / "sync_trace.json",
    )
    async_ = _profile(
        model,
        x_query,
        prefetch=True,
        path=args.output_dir / "async_trace.json",
    )
    assert sync.allclose(async_)
    model.clear()


if __name__ == "__main__":
    _main()
