# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Train Kumo Tabular on datasets sampled from its SCM prior."""

import argparse
import json
import os
from collections.abc import Callable
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.checkpoint import checkpoint as activation_checkpoint
from torch.utils.data import DataLoader
from train.kumo.tabular.loader import (
    COLUMN_BUCKETS,
    MAX_COLUMNS_LADDER,
    PriorDataset,
)
from train.kumo.tabular.optim import configure_optimizers
from train.kumo.tabular.prior import make_prior

from sdm import Task
from sdm._kernels.flash_attention import load_flash_attention
from sdm.models.kumo.tabular.block import KumoTabularTransformerBlock
from sdm.models.kumo.tabular.model import MODEL_KWARGS, _KumoTabular
from sdm.nn import SDPA, LogScale


class OneHotLinear(torch.nn.Linear):
    """Project one-hot class labels with a trainable weight and bias."""

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa: A002
        encoded = F.one_hot(input, num_classes=self.in_features)
        return super().forward(encoded.to(self.weight.dtype))


def save_checkpoint(path: Path, state: dict[str, Any]) -> None:
    """Atomically replace a checkpoint after writing it in full."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=Task,
        choices=(Task.classification, Task.regression),
        default=Task.classification,
    )
    parser.add_argument(
        "--size", choices=("small", "medium", "large"), default="small"
    )
    initialization = parser.add_mutually_exclusive_group()
    initialization.add_argument(
        "--init-checkpoint",
        type=Path,
        help="Initialize model weights; optimizer and schedule start fresh",
    )
    initialization.add_argument(
        "--resume",
        type=Path,
        help="Resume model and training state from this recovery checkpoint",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints"),
        help="Directory for recovery checkpoints, saved every 1000 updates",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("model.pt"),
        help="Checkpoint path",
    )
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=20,
        help="Number of optimizer updates; use 20 for a smoke test",
    )
    parser.add_argument("--min-rows", type=int, default=1024)
    parser.add_argument("--max-rows", type=int, default=1024)
    parser.add_argument(
        "--max-columns-ladder",
        type=json.loads,
        default=MAX_COLUMNS_LADDER,
        help="JSON [row upper bound, column cap] pairs; null means unbounded",
    )
    parser.add_argument("--min-train-size", type=float, default=0.3)
    parser.add_argument("--max-train-size", type=float, default=0.9)
    parser.add_argument("--global-batch-size", type=int, default=64)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument(
        "--keep-trivial-prob",
        type=float,
        default=0.5,
        help="Probability of allowing trivial datasets through quality checks",
    )
    parser.add_argument("--correlated-block-prob", type=float, default=0.3)
    parser.add_argument("--relabel-prob", type=float, default=0.1)
    parser.add_argument("--coarsen-prob", type=float, default=0.1)
    parser.add_argument("--high-cardinality-prob", type=float, default=0.05)
    parser.add_argument("--y-heavy-tail-prob", type=float, default=0.2)
    parser.add_argument(
        "--structured-missingness-prob", type=float, default=0.2
    )
    parser.add_argument("--mcar-missingness-prob", type=float, default=0.01)
    parser.add_argument(
        "--near-empty-missingness-prob", type=float, default=0.05
    )
    parser.add_argument(
        "--latent-driver-missingness-prob", type=float, default=0.03
    )
    parser.add_argument("--softmax-argmax-prob", type=float, default=0.15)
    parser.add_argument("--min-regression-train-std", type=float, default=0.1)
    parser.add_argument(
        "--mem-preserve-threshold",
        type=int,
        default=0,
        help="Row threshold for activation checkpointing and smaller "
        "microbatches; 0 disables both",
    )
    parser.add_argument("--warmup-proportion", type=float, default=0.02)
    parser.add_argument("--lr-end-factor", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=10.0)
    parser.add_argument(
        "--replay-small",
        action="store_true",
        help="Mix shorter datasets into long-context training",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Compile the training forward and backward",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--attention-backend",
        choices=("sdpa", "fa3"),
        default="sdpa",
        help="Use FlashAttention-3 with FP16/BF16",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Sampling workers per rank; 0 inline, -1 uses allocated CPUs",
    )
    parser.add_argument("--prefetch", type=int, default=2)
    parser.add_argument(
        "--precision", choices=("fp32", "fp16", "bf16"), default="fp32"
    )
    args = parser.parse_args()

    # Setup:
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    workers = args.workers
    if workers < 0:
        cpus = (
            len(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else (os.cpu_count() or 1)
        )
        workers = max(1, cpus // int(os.environ.get("LOCAL_WORLD_SIZE", 1)))
    if torch.cuda.is_available():
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    if world_size > 1:
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")

    # Training:
    try:
        if args.global_batch_size % (world_size * args.group_size):
            raise ValueError(
                "global_batch_size must be divisible by "
                "world_size * group_size"
            )
        if args.compile and (
            args.min_rows != args.max_rows or args.replay_small
        ):
            raise ValueError("--compile requires a fixed sequence length")
        batch_size = args.global_batch_size // world_size
        group_size = args.group_size
        if world_size > 1:
            counts = [None] * world_size
            dist.all_gather_object(counts, workers)
            if len(set(counts)) != 1:
                raise ValueError("Use equal worker counts on all ranks")
        # Retention keeps the newest files, so runs must not share a directory.
        if args.resume is None and any(args.checkpoint_dir.glob("step-*.pt")):
            raise FileExistsError(
                f"{args.checkpoint_dir} already contains recovery "
                "checkpoints; pass --resume or choose another --checkpoint-dir"
            )
        checkpoint = {}
        step = skipped = 0
        if args.resume is not None:
            checkpoint = torch.load(
                args.resume, map_location="cpu", weights_only=True
            )
            step, skipped = checkpoint["step"], checkpoint["skipped_steps"]
            if rank == 0:
                print(  # noqa: T201
                    f"Resuming {args.resume} at step {step}", flush=True
                )
        torch.manual_seed(args.seed + rank)
        model = _KumoTabular(
            num_classes=10 if args.task == Task.classification else 0,
            num_quantiles=999,
            device=device,
            **MODEL_KWARGS[args.size],
        ).train()
        for block in (model.row_embedding, model.icl_block):
            if isinstance(block.y_emb, torch.nn.Embedding):
                block.y_emb = OneHotLinear(
                    in_features=block.y_emb.num_embeddings,
                    out_features=block.y_emb.embedding_dim,
                    device=device,
                )
        for module in model.modules():
            if isinstance(module, LogScale):
                torch.nn.init.constant_(module.head_scale, 0.43)
        if args.init_checkpoint is not None:
            checkpoint = torch.load(
                args.init_checkpoint,
                map_location="cpu",
                weights_only=True,
            )
        if args.resume is not None or args.init_checkpoint is not None:
            model.load_state_dict(checkpoint["model"])
        if args.attention_backend == "fa3":
            if (
                device.type != "cuda"
                or torch.cuda.get_device_capability(device)[0] != 9
                or args.precision == "fp32"
            ):
                raise ValueError(
                    "--attention-backend fa3 requires Hopper FP16/BF16"
                )
            load_flash_attention()
        for module in model.modules():
            if isinstance(module, SDPA):
                module.backend = args.attention_backend
        optimizers, schedulers = configure_optimizers(
            model=model,
            lr=args.lr,
            warmup_proportion=args.warmup_proportion,
            max_steps=args.max_steps,
            lr_end_factor=args.lr_end_factor,
        )
        if args.resume is not None:
            for optimizer, state in zip(
                optimizers, checkpoint["optimizers"], strict=True
            ):
                optimizer.load_state_dict(state)
            for scheduler, state in zip(
                schedulers, checkpoint["schedulers"], strict=True
            ):
                scheduler.load_state_dict(state)
        checkpoint_activations = False

        def checkpoint_block(
            function: Callable[..., Any], *args: Any, **kwargs: Any
        ) -> Any:
            if checkpoint_activations and torch.is_grad_enabled():
                return activation_checkpoint(
                    function, *args, use_reentrant=False, **kwargs
                )
            return function(*args, **kwargs)

        if args.mem_preserve_threshold > 0:
            for block in model.modules():
                if isinstance(block, KumoTabularTransformerBlock):
                    block.forward = partial(checkpoint_block, block.forward)
        # Select the training entry point before DDP wraps the module call.
        model.forward = model.forward_train  # ty: ignore[invalid-assignment]
        if args.compile:
            max_context_length = min(
                max(int(args.max_rows * args.max_train_size), 1),
                args.max_rows - 1,
            )
            torch._dynamo.config.cache_size_limit = max(
                torch._dynamo.config.cache_size_limit,
                len(COLUMN_BUCKETS),
            )
            model.compile(fullgraph=True, dynamic=False)
        train_model = (
            DDP(
                model,
                device_ids=[local_rank] if device.type == "cuda" else None,
            )
            if world_size > 1
            else model
        )
        prior = make_prior(
            args.task,
            correlated_block_prob=args.correlated_block_prob,
            relabel_prob=args.relabel_prob,
            coarsen_prob=args.coarsen_prob,
            high_cardinality_prob=args.high_cardinality_prob,
            y_heavy_tail_prob=args.y_heavy_tail_prob,
            structured_missingness_prob=args.structured_missingness_prob,
            mcar_missingness_prob=args.mcar_missingness_prob,
            near_empty_missingness_prob=args.near_empty_missingness_prob,
            latent_driver_missingness_prob=args.latent_driver_missingness_prob,
            softmax_argmax_prob=args.softmax_argmax_prob,
        )
        batches = iter(
            DataLoader(
                PriorDataset(
                    prior=prior,
                    batch_size=batch_size,
                    group_size=group_size,
                    min_rows=args.min_rows,
                    max_rows=args.max_rows,
                    min_train_size=args.min_train_size,
                    max_train_size=args.max_train_size,
                    replay_small=args.replay_small,
                    # Restart sampling without replaying the initial stream.
                    seed=args.seed + step * world_size,
                    rank=rank,
                    world_size=world_size,
                    keep_trivial_prob=args.keep_trivial_prob,
                    min_regression_train_std=args.min_regression_train_std,
                    robust_regression_probe=args.y_heavy_tail_prob > 0,
                    max_columns_ladder=args.max_columns_ladder,
                ),
                batch_size=None,
                num_workers=workers,
                prefetch_factor=args.prefetch if workers else None,
                pin_memory=bool(workers and device.type == "cuda"),
                multiprocessing_context="spawn" if workers else None,
            )
        )
        dtype = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }[args.precision]
        scaler = torch.amp.GradScaler(
            device.type, enabled=args.precision == "fp16"
        )

        if args.resume is not None and checkpoint["scaler"]:
            scaler.load_state_dict(checkpoint["scaler"])
        del checkpoint

        # Training loop:
        consecutive_skips = 0
        while step < args.max_steps:
            groups = next(batches)
            num_rows = groups[0].x.size(1)
            checkpoint_activations = num_rows > args.mem_preserve_threshold

            # Shrink microbatches for large datasets; gradients accumulate.
            microbatch_size = group_size
            if (
                args.mem_preserve_threshold > 0
                and num_rows >= 2 * args.mem_preserve_threshold
            ):
                microbatch_size = max(
                    1, group_size * args.mem_preserve_threshold // num_rows
                )
                # Keep equally sized microbatches within each sampling group.
                while group_size % microbatch_size:
                    microbatch_size -= 1

            for optimizer in optimizers:
                optimizer.zero_grad(set_to_none=True)
            loss_sum = torch.zeros((), device=device)
            processed = 0
            # Accumulate all groups before a single optimizer step.
            for group in groups:
                train_size = group.train_size
                for start in range(0, group_size, microbatch_size):
                    batch = slice(start, start + microbatch_size)
                    x = group.x[batch].to(device, non_blocking=True)
                    y = group.y[batch].to(device, non_blocking=True)
                    y_context, y_query = y[:, :train_size], y[:, train_size:]
                    categorical_mask = group.categorical_mask[batch].to(
                        device, non_blocking=True
                    )
                    padding = group.max_column_width - x.size(-1)
                    if args.compile and padding:
                        x = F.pad(x, (0, padding))
                        categorical_mask = F.pad(
                            categorical_mask, (0, padding)
                        )
                    y_input = y_context
                    context_lengths = None
                    if args.compile:
                        y_input = F.pad(
                            y_context,
                            (0, max_context_length - train_size),
                        )
                        context_lengths = torch.full(
                            (x.size(0),),
                            train_size,
                            dtype=torch.int32,
                            device=device,
                        )
                    processed += x.size(0)
                    # Suppress gradient sync until the final microbatch.
                    sync = (
                        train_model.no_sync()
                        if isinstance(train_model, DDP)
                        and processed < batch_size
                        else nullcontext()
                    )
                    with sync:
                        with torch.autocast(
                            device_type=device.type,
                            dtype=dtype,
                            enabled=args.precision != "fp32",
                        ):
                            out = train_model(
                                x=x,
                                y=y_input,
                                categorical_mask=categorical_mask,
                                train_size=context_lengths,
                                num_columns=group.num_columns[batch].to(
                                    device, non_blocking=True
                                ),
                            )
                        if args.compile:
                            out = out[:, train_size:]
                        # Compute the loss in FP32, outside autocast.
                        out = out.float()
                        if args.task == Task.classification:
                            present = F.one_hot(
                                y_context, num_classes=out.size(-1)
                            ).any(dim=-2)
                            out = out.masked_fill(
                                ~present.unsqueeze(-2),
                                torch.finfo(out.dtype).min,
                            )
                            loss = F.cross_entropy(
                                out.flatten(0, -2),
                                y_query.flatten(),
                            )
                        else:
                            diff = y_query.unsqueeze(-1) - out
                            levels = torch.linspace(
                                0.001, 0.999, 999, device=device
                            )
                            loss = torch.maximum(
                                levels * diff, (levels - 1) * diff
                            ).mean()
                        # Equal dataset weights; backward frees this graph.
                        weight = x.size(0) / batch_size
                        scaler.scale(loss * weight).backward()
                    loss_sum += loss.detach() * weight

            # Keep one scale for all microbatches; unscale before clipping.
            for optimizer in optimizers:
                scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=args.grad_clip,
            )
            # Muon and AdamW must either both update or both skip.
            if not norm.isfinite():
                scaler.update()
                skipped += 1
                consecutive_skips += 1
                if consecutive_skips >= 100:
                    raise FloatingPointError(
                        "100 consecutive nonfinite updates"
                    )
                continue
            for optimizer in optimizers:
                scaler.step(optimizer)
            scaler.update()
            for scheduler in schedulers:
                scheduler.step()
            step += 1
            consecutive_skips = 0
            loss = loss_sum
            if world_size > 1:
                dist.all_reduce(loss)
                loss /= world_size
            if rank == 0:
                print(  # noqa: T201
                    f"step={step}/{args.max_steps} loss={float(loss):.6f} "
                    f"skipped={skipped}"
                )

            if step % 1000 == 0 or step == args.max_steps:
                if rank == 0:
                    save_checkpoint(
                        path=args.checkpoint_dir / f"step-{step:012d}.pt",
                        state={
                            "model": model.state_dict(),
                            "optimizers": [
                                opt.state_dict() for opt in optimizers
                            ],
                            "schedulers": [
                                sch.state_dict() for sch in schedulers
                            ],
                            "scaler": scaler.state_dict(),
                            "step": step,
                            "skipped_steps": skipped,
                        },
                    )
                    for path in sorted(args.checkpoint_dir.glob("step-*.pt"))[
                        :-5
                    ]:
                        path.unlink()
                if world_size > 1:
                    dist.barrier()

        if rank == 0:
            state = model.state_dict()
            save_checkpoint(
                path=args.output,
                state={
                    "model": state,
                    "task": args.task.value,
                    "size": args.size,
                    "precision": args.precision,
                },
            )
            for prefix in ("row_embedding.y_emb", "icl_block.y_emb"):
                bias = state.pop(f"{prefix}.bias", None)
                if bias is not None:
                    state[f"{prefix}.weight"] = (
                        state[f"{prefix}.weight"].T.contiguous() + bias
                    )
            save_checkpoint(
                path=args.output.with_stem(f"{args.output.stem}-sdm"),
                state=state,
            )
        if world_size > 1:
            dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
