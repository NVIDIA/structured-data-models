# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real process-group parity, including nonzero trained-like residuals."""

import os
import subprocess
import sys
from pathlib import Path
from typing import Literal

import pytest
import torch
import torch.distributed as dist

from sdm.cache import Cache
from sdm.models.kumo.tabular.icl import ICLBlock as KumoICL
from sdm.models.tabiclv2.icl import ICLBlock as RelationalICL
from sdm.nn import SDPA
from sdm.nn.context_parallel import (
    cached_context,
    context_parallel,
    context_parallel_attention,
)


def _worker(
    rank: int,
    world: int,
    rendezvous: str,
    device_type: str = "cpu",
    dtype: torch.dtype = torch.float32,
    kernel: Literal["efficient", "flash", "efficient_fp32"] = "efficient",
) -> None:
    torch.set_num_threads(1)
    if device_type == "cuda":
        torch.cuda.set_device(rank)
        torch.set_default_device(f"cuda:{rank}")
    dist.init_process_group(
        "nccl" if device_type == "cuda" else "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=world,
    )
    atol, rtol = (0.003, 0.02) if dtype == torch.bfloat16 else (3e-6, 3e-5)
    try:
        with (
            torch.inference_mode(),
            torch.autocast(
                device_type, dtype=dtype, enabled=dtype != torch.float32
            ),
        ):
            # Identical replicated queries/model weights are the CP contract.
            torch.manual_seed(32)
            for length in (0, 1, 7, 17):
                for kv_heads in (1, 4):
                    q = torch.randn(2, 3, 4, 8)
                    k = torch.randn(1, length, kv_heads, 8)
                    v = torch.randn(2, length, kv_heads, 8)
                    native = SDPA(4, kv_heads)(q, k, v)
                    actual = context_parallel_attention(
                        q,
                        k.tensor_split(world, -3)[rank],
                        v.tensor_split(world, -3)[rank],
                        group=dist.group.WORLD,
                        kernel=kernel,
                    )
                    torch.testing.assert_close(
                        actual, native, atol=atol, rtol=rtol, check_dtype=False
                    )
            for family, kv_heads in (
                ("tabular", None),
                ("tabular", 1),
                ("relational", None),
            ):
                kwargs = {
                    "num_classes": 3,
                    "out_channels": 3,
                    "channels": 32,
                    "num_layers": 3,
                    "num_heads": 4,
                }
                model = (
                    KumoICL(**kwargs, num_key_value_heads_for_query=kv_heads)
                    if family == "tabular"
                    else RelationalICL(**kwargs, norm_bias=True)
                )
                # Fresh modules initialize output projections to zero; replace
                # those identities so a broken attention cannot pass unnoticed.
                for param in model.parameters():
                    param.uniform_(-0.2, 0.2)
                model.eval()
                x, query = torch.randn(2, 7, 32), torch.randn(2, 5, 32)
                y = torch.randint(3, (2, 7))
                native_cache = Cache()
                model(x.clone(), y, cache=native_cache)
                native_cache.freeze()
                expected = model(query.clone(), y[..., :0], cache=native_cache)
                cache = Cache()
                with context_parallel(dist.group.WORLD, kernel=kernel):
                    model(x.clone(), y, cache=cache)
                    cache.freeze()
                    actual = model(query.clone(), y[..., :0], cache=cache)
                    torch.testing.assert_close(
                        actual, expected, atol=atol, rtol=rtol
                    )
                    actual_cache_bytes = cache.size()
                    assert actual_cache_bytes < native_cache.size()
                    for key, value in cache.items():
                        if key.endswith(".context_parallel"):
                            continue
                        for tensor in value:
                            assert (
                                tensor.untyped_storage().nbytes()
                                == tensor.numel() * tensor.element_size()
                            )
                    with (
                        cached_context(cache, "icl_block.layer0"),
                        pytest.raises(NotImplementedError, match="masks"),
                    ):
                        SDPA(4)(
                            q,
                            k,
                            v,
                            attn_mask=torch.ones(3, 17, dtype=torch.bool),
                        )
                with pytest.raises(RuntimeError, match="topology"):
                    model(query.clone(), y[..., :0], cache=cache)
                with (
                    context_parallel(dist.group.WORLD),
                    pytest.raises(RuntimeError, match="Fit the cache"),
                ):
                    model(query.clone(), y[..., :0], cache=native_cache)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world", [2, 4])
def test_distributed_context_attention(tmp_path: Path, world: int) -> None:
    _run_workers(tmp_path, world, "cpu", torch.float32, "efficient")


def test_requires_inference() -> None:
    with (
        pytest.raises(RuntimeError, match="inference"),
        context_parallel(dist.group.WORLD),
    ):
        pass


@pytest.mark.parametrize("world", [2, 4])
@pytest.mark.parametrize(
    ("kernel", "dtype"),
    [
        ("efficient", torch.float32),
        ("efficient", torch.bfloat16),
        ("flash", torch.bfloat16),
        ("efficient_fp32", torch.bfloat16),
    ],
)
def test_cuda_distributed_context_attention(
    tmp_path: Path, world: int, dtype: torch.dtype, kernel: str
) -> None:
    if torch.cuda.device_count() < world:
        pytest.skip(f"Requires {world} CUDA devices")
    _run_workers(tmp_path, world, "cuda", dtype, kernel)


def _run_workers(
    tmp_path: Path, world: int, device: str, dtype: torch.dtype, kernel: str
) -> None:
    # Pytest importlib names can collide with Linux's stdlib `test` package.
    # Launch this file directly instead of pickling a pytest-module function.
    root = str(Path(__file__).resolve().parents[2])
    subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--local-addr=127.0.0.1",
            f"--nproc-per-node={world}",
            __file__,
            f"file://{tmp_path / 'rendezvous'}",
            device,
            str(dtype).removeprefix("torch."),
            kernel,
        ],
        env={
            **os.environ,
            "PYTHONPATH": root + os.pathsep + os.environ.get("PYTHONPATH", ""),
        },
        check=True,
    )


if __name__ == "__main__":
    _worker(
        int(os.environ["LOCAL_RANK"]),
        int(os.environ["WORLD_SIZE"]),
        sys.argv[1],
        sys.argv[2],
        getattr(torch, sys.argv[3]),
        sys.argv[4],
    )
