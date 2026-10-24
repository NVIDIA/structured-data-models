# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reuse a sharded KumoTabular context for held-out synthetic queries.

Run with torchrun --standalone --nproc-per-node=2 --max-restarts=0
examples/tabular/context_sharding.py --precision float32.
Use --precision float16 for opt-in FP16 neural execution; model weights and
numerical inputs remain FP32. Pretrained weights must be available or will be
downloaded by KumoTabular. This small example demonstrates the API, not a
speedup or a predictive-quality benchmark. Use a finite external job timeout.

On failure, let torchrun retire the workers. Do not try collective clear/close
from a one-rank exception handler or reuse a failed executor.
"""

import argparse
import os
from datetime import timedelta

import torch
import torch.distributed as dist

import sdm
from sdm.execution import tabular_execution


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--precision", choices=("float32", "float16"), default="float32"
    )
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dist.init_process_group("nccl", timeout=timedelta(minutes=5))
    group = dist.group.WORLD
    assert group is not None
    control = dist.new_group(backend="gloo", timeout=timedelta(minutes=5))
    assert isinstance(control, dist.ProcessGroup)
    root = dist.get_rank() == 0

    # All ranks load the same stock checkpoint before borrowing the model.
    model = sdm.models.KumoTabular(
        task="regression", size="large", device=device
    ).eval()
    run = tabular_execution(
        model,
        task="regression",
        group=group,
        control_group=control,
        precision=args.precision,
        owner_local_frontend=True,
        shard_row_workspace=True,
        fit_head_gather_window=2,
    )

    x_context = y_context = x_query = batches = None
    if root:
        generator = torch.Generator().manual_seed(42)
        features = torch.randn(576, 8, generator=generator).to(device)
        # Fit preprocessing only on context; the last 64 rows are held out.
        x_context, x_query = features[:512], features[512:]
        y_context = x_context[:, :1] + x_context[:, 1:2].sin()
        batches = list(x_query.split(16))

    # All ranks call every method. The executor owns the precision scope.
    # Use no_grad, not inference_mode: fitted caches need version counters.
    with torch.no_grad():
        generator = (
            torch.Generator(device=device).manual_seed(42) if root else None
        )
        run.fit(x_context, y_context, generator=generator)
        predictions = run.predict_batches(batches)
        run.clear()
        generator = (
            torch.Generator(device=device).manual_seed(42) if root else None
        )
        run.fit(x_context, y_context, generator=generator)
        prediction = run.predict(x_query)
        run.close()

    # Healthy completion only: caller-owned groups outlive the model lease.
    dist.destroy_process_group(control)
    dist.destroy_process_group()
    if root:
        assert predictions is not None
        assert prediction is not None
        print("Original batch output shapes:", [p.shape for p in predictions])
        print("Refitted output shape:", prediction.shape)  # [64, 999]


if __name__ == "__main__":
    main()
