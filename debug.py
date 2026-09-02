# ruff: noqa: D103, T201

import argparse
import time

import torch

import sdm
import sdm.models.kumo.tabular.row_embedding as row_embedding


def gib(num_bytes: int) -> float:
    return num_bytes / 1024**3


def report(name: str) -> None:
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    print(
        f"{name:>34s}: "
        f"alloc={gib(torch.cuda.memory_allocated()):6.2f} GiB, "
        f"reserved={gib(torch.cuda.memory_reserved()):6.2f} GiB, "
        f"peak={gib(torch.cuda.max_memory_allocated()):6.2f} GiB, "
        f"free={gib(free):6.2f}/{gib(total):.2f} GiB",
        flush=True,
    )


def add_memory_hooks(module: torch.nn.Module, name: str) -> None:
    def pre_hook(
        _module: torch.nn.Module,
        _args: tuple[object, ...],
    ) -> None:
        report(f"before {name}")

    def hook(
        _module: torch.nn.Module,
        _args: tuple[object, ...],
        _output: object,
    ) -> None:
        report(f"after {name}")

    module.register_forward_pre_hook(pre_hook)
    module.register_forward_hook(hook)


def add_kumo_hooks(model: sdm.models.KumoTabular) -> None:
    kumo = next(iter(model.models.values()))
    add_memory_hooks(kumo.cell_embedding, "cell_embedding")
    add_memory_hooks(kumo.row_embedding, "row_embedding")
    add_memory_hooks(kumo.row_project, "row_project")
    add_memory_hooks(kumo.icl_block, "icl_block")

    for i, block in enumerate(kumo.row_embedding.col_blocks):
        add_memory_hooks(block, f"row_embedding.col_blocks.{i}")
    for i, block in enumerate(kumo.row_embedding.row_blocks):
        add_memory_hooks(block, f"row_embedding.row_blocks.{i}")
    for i, layer in enumerate(kumo.icl_block.layers):
        add_memory_hooks(layer, f"icl_block.layers.{i}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-memory", action="store_true")
    parser.add_argument("--col-batch-size-limit", type=int)
    parser.add_argument("--row-batch-size-limit", type=int)
    args = parser.parse_args()

    if args.col_batch_size_limit is not None:
        row_embedding._COL_BATCH_SIZE_LIMIT = args.col_batch_size_limit
    if args.row_batch_size_limit is not None:
        row_embedding._ROW_BATCH_SIZE_LIMIT = args.row_batch_size_limit

    model = sdm.models.TabICLv2(
        task="regression",
        # size="large",
        device="cuda",
    )
    if args.profile_memory:
        add_kumo_hooks(model)

    x_context = torch.randn(50_000, 170, device="cuda")
    y_context = torch.randn(50_000, 1, device="cuda")
    x_query = torch.randn(25_000, 170, device="cuda")

    torch.cuda.reset_peak_memory_stats()
    if args.profile_memory:
        report("after inputs")

    start = time.perf_counter()
    try:
        with torch.amp.autocast("cuda", torch.float16, enabled=True):
            if args.profile_memory:
                report("before fit")
            fit_start = time.perf_counter()
            model.fit(x_context, y_context)
            torch.cuda.synchronize()
            fit_elapsed = time.perf_counter() - fit_start
            if args.profile_memory:
                report("after fit")

            if args.profile_memory:
                print("--------", flush=True)

            if args.profile_memory:
                report("before predict")
            predict_start = time.perf_counter()
            model.predict(x_query)
            torch.cuda.synchronize()
            predict_elapsed = time.perf_counter() - predict_start
            if args.profile_memory:
                report("after predict")
    except torch.OutOfMemoryError as e:
        report("OOM")
        raise e
    finally:
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start
    print(
        f"elapsed={elapsed:.2f}s, "
        f"fit={fit_elapsed:.2f}s, "
        f"predict={predict_elapsed:.2f}s, "
        f"peak={gib(torch.cuda.max_memory_allocated()):.2f} GiB",
        flush=True,
    )


if __name__ == "__main__":
    main()
