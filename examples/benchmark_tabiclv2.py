r"""Inference acceleration benchmark for :class:`TabICLv2`.

Sweeps NVIDIA-recommended inference configurations over representative
in-context learning workloads and reports wall-clock latency,
table-stream cost (amortized over a stream of fresh table shapes,
including recompiles), peak memory (the cell's maximum, so one-shot cells
include their up-to-1.25x-row stream tables, and bucketed cells their
bucket-padded shapes - up to 1.375x rows and 1.125-1.5x columns), and
accuracy versus an fp32 reference.

What to expect from each precision (measured on GB200 with a torch 2.14
nightly, CUDA 13.4 and cuDNN 9.26; 8192x64-row one-shot classification
unless noted):

* ``fp32`` (``ieee`` matmuls, ``c0``): the accuracy reference; slowest
  (~106 ms).
* ``tf32`` (``c1``): 1.4x for fp32 pipelines with negligible drift; the
  safe default (``torch.set_float32_matmul_precision("high")``, 72 ms).
* ``bf16`` (full-cast, ``c3``): 4x and half the peak memory (26 ms),
  passing all accuracy gates; ~5.8x when combined with
  ``compile(fullgraph=True, dynamic=True)`` (``c6``, 18.4 ms), and ~4x
  on small launch-bound tables with ``mode="reduce-overhead"`` (``c16``,
  4.2 vs 16.4 ms). Static compilation (``c5``) reaches the same steady
  state (17.6 ms) but pays one automatic-dynamic recompile (~30 s) on
  the second table shape, so keep ``dynamic=True`` for in-context
  learning. The autocast variant of this recipe (``c12``) measures
  within noise of ``c6`` one-shot and ~7.5 ms per cached predict on
  the fit/predict route (the fit itself is reported separately as
  ``fit_s``, ~15 ms large); ``examples/tabiclv2/quickstart.py`` ships
  its ``float16`` counterpart, measured within noise (see ``fp16``).
* ``fp8`` (Transformer Engine 2.19, per-tensor scaling, ``c9``):
  measured *slower* than bf16 here - ~25 ms small / ~30 ms large
  against ``c3``'s ~16-19 / ~26-31 ms - at this model's GEMM sizes
  (128-1536 channels) quantization overhead cancels the gain. Accuracy
  gates pass. Shapes are constrained: feature dims must be multiples
  of 16 and leading-dimension products multiples of 8, so arbitrary
  table sizes need padding.
* ``mxfp8`` (block-scaled fp8, Blackwell, ``c10``): tracks ``fp8``
  (~25 ms small / ~30-42 ms large, the large cell flagged ``NOISY`` in
  one of two runs) - no win over bf16 at these GEMM sizes; same shape
  constraints as fp8.
* ``nvfp4`` (4-bit block-scaled, Blackwell, ``c11``): slowest of the
  family (~28 ms small / ~32 ms large) and the only configuration to
  fail an accuracy gate, on the 256x8-row small workload: pooled over
  five seeded tables its median per-row logit shift is 24% of the
  decision margin (threshold 10%), with top-1 agreement at 99.7%; the
  large workload passes (8%). Built for much larger GEMMs - not
  recommended for TabICLv2.

* ``fp16`` (``c13``/``c14``): measured within noise of bf16 everywhere
  (26.0 ms large eager; ``c14`` tracks ``c6``); bf16 keeps the recipe
  slot for its wider exponent range.

Beyond precision, the shape dimension (all measured on GB200):

* **Fresh-shape tax**: whenever attention runs on the cuDNN SDPA
  backend (the bf16/fp16 cells, eager or compiled: the cells with a
  ``True`` priority field pin it first, and torch 2.14's default order
  also selects it for these shapes), every new (rows, columns)
  combination costs ~200 ms of host-side cuDNN attention-plan building
  on its first call (~87% of first-call CPU time);
  ``CUDA_MODULE_LOADING=EAGER`` and expandable segments do not help.
  That figure is a warm-CUDA-JIT-cache number: the plan build
  runtime-compiles the attention kernel and the driver caches the JIT
  result on disk (``CUDA_CACHE_PATH``). With the cache disabled
  (``CUDA_CACHE_DISABLE=1`` in the environment, inherited by every
  cell) nothing is kept and each fresh shape costs ~1.5 s small /
  ~1.9 s large (CPU-bound) in eager, whole-model and regional cells
  alike. With the cache enabled the surcharge belongs to the cache
  directory, not to the process or container session: the first process
  to serve a table-size class on cuDNN attention against a directory
  that lacks that class's kernels (the small- and large-table attention
  kernels are distinct compiles) pays a one-time surcharge on its cold
  first call and its first fresh table - +~250 ms small; +~1.2 s large
  on an empty directory (a first deployment, a driver upgrade), +~0.7 s
  once a small-table process has written the shared kernels - and adds
  that class's files (``c6`` small 13-15, large 6-15), even when earlier
  processes in the same session already warmed the other class; every
  later process against that directory, in the same or a new container
  session, is warm for that class (no files added, first table within
  ~15 ms of its pass median) and later shapes within a process are warm,
  so a persistent ``CUDA_CACHE_PATH`` carries the warm state across
  sessions. Each record's ``environment`` says whether the cache is
  disabled and how many files the cell added to the directory
  (``cuda_cache_files_added``). That count is raw (the directory also
  receives torch's own runtime-JIT'd kernels: on an empty directory
  even the eager fp32 ``c0`` adds a few files with a flat stream), so
  the summary flags a cell ``JIT`` when its pass-1 first table exceeds
  3x its pass median or, for an unbucketed cell whose attention can
  reach the cuDNN backend, when it added files or ran with the cache
  disabled, because a plan-build miss inflates ``stream=`` (``stream2=``
  is the steady value once the cache holds the shapes); bucketed cells
  absorb the miss in their bucket warmup, outside the timed stream, so
  for them only the first-table rule applies. The fp32 cells
  never reach cuDNN attention and pay under ~1 ms for a fresh shape
  (first call minus steady p50 at the same shape: ~0.7 ms small, ~0.2
  ms large); the ~16 ms between ``c0``'s large stream= and p50= is the
  jittered tables being up to 1.25x larger, not the shapes being new.
  The tax is specific to that backend: pinning
  ``sdpa_kernel([FLASH_ATTENTION, EFFICIENT_ATTENTION, MATH],
  set_priority=True)`` (``c21``, the ``c6`` recipe without cuDNN
  attention) removes it with every accuracy gate passing, at 19.5 /
  9.0 ms one-shot p50 (~8% above ``c6`` on large - 18.1 ms in the same
  run - and within noise of it on small, 9.4 ms). What remains is a
  per-process warm-up over the first fresh shapes a process serves,
  whichever shapes they are: the first fresh table costs ~2x the p50
  (~20 ms small, ~33-38 ms large) and the surcharge fades over the next
  six or so, while revisiting a shape costs the on-grid p50 (small) or
  ~20-25 ms by table size (large). The two stream passes therefore
  differ (they agree for every cuDNN-attention cell): ``stream=`` (pass
  1) reads ~14 ms small / ~27 ms large and ``stream2=`` ~10 / ~24 ms
  (per-table medians 13.5 / 26 and 10 / 22.5 ms); the same priority
  under ``c17``-class regional compilation measures 19.3 ms p50 and
  ~27 / ~24 ms per fresh large table. On the large stream that ties
  ``c20`` (~27-28 ms) on the first pass and beats it by ~12% at steady
  state, with no padding or bucket warmup. Bucketing and regional
  compilation are the mitigations when cuDNN attention is kept.
* **Shape bucketing** (``c15``/``c16``/``c18``): pad train rows (masked
  exactly via ``seqused_train``), columns (masked via ``seqused_cols``),
  and test rows (extra outputs sliced away) up to a geometric bucket
  grid, so a stream of fresh tables revisits a small set of warm shapes.
  Measured warm-stream cost per fresh table (per-table median): 5.1 ms
  small / 43 ms large (``c16``) vs ~180-200 ms unbucketed on cuDNN
  attention (``c6``/``c17``) - ~35x small and ~4x large on the stream
  metric - after a one-time bucket warmup reported as
  ``bucket_warmup_s``. Against the unbucketed flash-priority ``c21``
  (~14 -> ~10 ms small / ~27 -> ~24 ms large per fresh table over its
  two passes) bucketing keeps a 2-2.7x win on small tables and loses on
  large ones. The
  first pass over fresh tables starts with one table ~0.3-2 ms above the
  rest in every bucketed recipe (``c15``/``c16``/``c18``, not only under
  ``mode="reduce-overhead"``), so on small tables its mean sits up to
  ~0.3 ms/table above the median (``c15``/``c16``; the ``c18`` first
  table stays within that pass's dispersion, so its mean sits at the
  median); the per-table times are recorded (``table_stream_tables_s``)
  and the second pass (``table_stream_fresh2_per_table_s``) is
  outlier-free. Padding correctness is gated per cell on an off-grid
  table against the same configuration's unpadded output
  (``padding_gate``).
* **Regional compilation** (``c17``/``c18``/``c19``): compiling the
  repeated transformer blocks individually instead of the whole backbone
  cuts ``cold_first_call_s`` from ~45-58 s (``c6``/``c16``) to ~15-21 s
  (``c17``/``c18``) and the masked-family ``bucket_warmup_s`` from
  ~30-44 s (``c16``) to ~11-17 s (``c18``), and measures the same
  steady state on large tables within noise (``c17`` 17.9 vs ``c6``
  18.4 ms). The per-fresh-shape stream cost is the same either way
  (~170-220 ms on cuDNN attention, ~24-27 ms under the flash priority):
  with ``dynamic=True`` neither recipe recompiles on a new shape, the
  cost is the attention backend's, as above. The trade-off is per-block
  dispatch overhead on launch-bound paths: small-table one-shot and
  cached-predict calls measure ~1-3 ms slower than whole-model
  compilation. Regional recipes also raise
  ``torch._dynamo.config.recompile_limit`` (default 8) to 64: every
  block instance shares one ``forward`` code object, and the fit/predict
  route's KV-cache record/replay specializations push the shared entry
  past 8, which under ``fullgraph=True`` raises
  ``FailOnRecompileLimitHit`` rather than falling back to eager. The
  entries are per-site specializations (input rank, ``attn_mask`` /
  ``seqused_key_value`` presence, cached vs fresh key/value, block
  width), not shapes: one process serving every shipped workload and
  task through both routes with bucketing fills 40 of the 64 for
  ``TransformerBlock.forward`` and 17 for
  ``InducedTransformerBlock.forward`` (GB200, torch 2.14; identical
  with the variable-length path on), on-grid shapes add none, and each
  further input rank or task adds roughly 15 per shared entry - size the
  limit for the deployment's rank/task mix instead of reusing 64.
  Regional compilation is supported with the default compile mode only:
  the first ICL layer hands its output to the later blocks' ``out=``
  buffers, which CUDA graph trees (``mode="reduce-overhead"``) reject,
  so the driver refuses that combination.
* **cuDNN variable-length attention** (``c20``): ``c18`` plus native
  padding-mask attention for the padded train-row key/value streams
  (``seqused_train``; column padding keeps the boolean key mask in row
  attention) - requires the optional ``cudnn`` extra,
  ``nvidia-cudnn-frontend``. Measured: large fresh-table stream 42.5 ->
  28 ms (1.5x; 10 execution graphs, 0 fallbacks); small tables ~1 ms
  slower per fresh table (13-14 -> 14-15 ms) from per-call graph
  execution overhead; the on-grid p50 is unchanged (~18 ms) because
  exact-fit tables pass no ``seqused_*`` and never reach the path.
  Both sides of that comparison run under the cuDNN-first SDPA
  priority; the unbucketed flash-priority ``c21`` serves the same large
  stream at ~24 ms per fresh table once warm (~27 ms over a process's
  first eight fresh tables).
  The cell aborts rather than publishing boolean-mask numbers under the
  variable-length label when the package is missing or a served shape
  falls back to the mask.
* **Compile-cache artifacts** (measured with a separate script, not
  part of this grid): ``torch.compiler.save_cache_artifacts`` after a
  warm run and ``load_cache_artifacts`` at boot cut cold compile to
  ~19 s (whole-model) or ~8 s (regional) with bitwise-identical
  outputs, and an AOTInductor ``.pt2`` package of the backbone loads in
  ~0.4 s for zero-JIT deployments.

The fp8/mxfp8/nvfp4 configurations require the optional
``transformer_engine`` package (available in NVIDIA NGC containers) and
swap eligible ``torch.nn.Linear`` modules for ``te.Linear``.

Each cell runs in a fresh subprocess so that inductor caches, CUDA-graph
pools, and allocator state cannot leak between configurations::

    python examples/benchmark_tabiclv2.py --out bench.json \
        --budget-minutes 110 --configs c0-fp32,c3-bf16-full,c10-mxfp8
"""

import argparse
import importlib
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from contextlib import ExitStack
from typing import Any, cast

import torch
from torch import Tensor
from torch.nn.attention import SDPBackend, sdpa_kernel

import sdm
from sdm import Recipe
from sdm.cache import Cache, KVCacheEntry
from sdm.models import TabICLv2
from sdm.nn import (
    InducedTransformerBlock,
    QASSMax,
    TransformerBlock,
    cudnn_varlen_stats,
    enable_cudnn_varlen,
)
from sdm.processing.execution import RecipeExecution

# `sdpa_kernel` priorities a configuration's second field can select:
# `False` keeps torch's default backend order, `True` pins the cuDNN-first
# order the bf16/fp16 recipes were measured with, and a tuple of
# `SDPBackend` names pins that order. Leaving CUDNN_ATTENTION out removes
# the per-fresh-shape plan-building cost described in the module docstring.
CUDNN_SDPA_PRIORITY = (
    "CUDNN_ATTENTION",
    "FLASH_ATTENTION",
    "EFFICIENT_ATTENTION",
    "MATH",
)
FLASH_SDPA_PRIORITY = ("FLASH_ATTENTION", "EFFICIENT_ATTENTION", "MATH")


def sdpa_backends(
    priority: bool | tuple[str, ...],
) -> list[SDPBackend] | None:
    """Resolve a configuration's SDPA priority field to backends."""
    if priority is False:
        return None
    if priority is True:
        priority = CUDNN_SDPA_PRIORITY
    return [getattr(SDPBackend, name) for name in priority]


CONFIGS = {
    # name: (precision, sdpa_priority, compile_kwargs, te_recipe) with an
    # optional fifth element enabling shape-bucketed padding and an
    # optional sixth enabling cuDNN variable-length attention.
    "c0-fp32": ("fp32", False, None, None),
    "c1-tf32": ("tf32", False, None, None),
    "c2-bf16-autocast": ("bf16-autocast", False, None, None),
    "c3-bf16-full": ("bf16-full", False, None, None),
    "c4-bf16-cudnn-sdpa": ("bf16-full", True, None, None),
    "c5-compile": ("bf16-full", True, {"fullgraph": True}, None),
    "c6-compile-dynamic": (
        "bf16-full",
        True,
        {"fullgraph": True, "dynamic": True},
        None,
    ),
    # c6 without cuDNN attention: the flash kernel serves fresh shapes
    # with no per-shape plan building (see the module docstring).
    "c21-compile-dynamic-flash": (
        "bf16-full",
        FLASH_SDPA_PRIORITY,
        {"fullgraph": True, "dynamic": True},
        None,
    ),
    "c7-compile-ro": (
        "bf16-full",
        True,
        {"fullgraph": True, "dynamic": True, "mode": "reduce-overhead"},
        None,
    ),
    "c8-compile-ma": (
        "bf16-full",
        True,
        {
            "fullgraph": True,
            "dynamic": True,
            "mode": "max-autotune-no-cudagraphs",
        },
        None,
    ),
    # The examples/tabiclv2/quickstart.py-style recipe: autocast
    # (TableTensor-friendly) combined with dynamic fullgraph compilation.
    # Measured in bf16; quickstart.py ships the float16 autocast variant,
    # which measures within noise of bf16 (see the fp16 cells).
    "c12-autocast-compile": (
        "bf16-autocast",
        False,
        {"fullgraph": True, "dynamic": True},
        None,
    ),
    # Transformer Engine low-precision GEMMs on a bf16 base model.
    "c9-fp8": ("bf16-full", True, None, "fp8"),
    "c10-mxfp8": ("bf16-full", True, None, "mxfp8"),
    "c11-nvfp4": ("bf16-full", True, None, "nvfp4"),
    # fp16 counterparts of the bf16 full-cast cells (c3/c6).
    "c13-fp16-full": ("fp16-full", False, None, None),
    "c14-fp16-compile-dynamic": (
        "fp16-full",
        True,
        {"fullgraph": True, "dynamic": True},
        None,
    ),
    # Shape-bucketed serving: pad train rows (masked via `seqused_train`)
    # and test rows (extra outputs discarded) up to a geometric bucket
    # grid so fresh tables reuse warm shapes. The fifth tuple element
    # enables bucketing in the stream, timing, and accuracy routes.
    "c15-bucket-compile-dynamic": (
        "bf16-full",
        True,
        {"fullgraph": True, "dynamic": True},
        None,
        True,
    ),
    "c16-bucket-compile-ro": (
        "bf16-full",
        True,
        {"fullgraph": True, "dynamic": True, "mode": "reduce-overhead"},
        None,
        True,
    ),
    # Regional compilation: compile the repeated transformer blocks
    # individually instead of the whole backbone. Dynamo reuses compiled
    # code across the structurally identical blocks, cutting cold compile
    # time and removing the whole-graph dynamic-shape recompile.
    "c17-regional-compile": (
        "bf16-full",
        True,
        {"fullgraph": True, "dynamic": True, "regional": True},
        None,
    ),
    # The full serving recipe: regional compilation + bucketed shapes.
    "c18-bucket-regional": (
        "bf16-full",
        True,
        {"fullgraph": True, "dynamic": True, "regional": True},
        None,
        True,
    ),
    # The quickstart.py-style autocast recipe with regional compilation.
    "c19-autocast-regional": (
        "bf16-autocast",
        False,
        {"fullgraph": True, "dynamic": True, "regional": True},
        None,
    ),
    # c18 plus cuDNN variable-length attention for the padded train-row
    # key/value streams (column padding keeps the boolean key mask;
    # requires the optional nvidia-cudnn-frontend package). The sixth
    # tuple element enables the path.
    "c20-bucket-regional-varlen": (
        "bf16-full",
        True,
        {"fullgraph": True, "dynamic": True, "regional": True},
        None,
        True,
        True,
    ),
}

WORKLOADS = {
    # name: (batch_shape, num_rows, num_columns, num_train_rows)
    "small": ((), 256, 8, 192),
    "medium": ((), 2048, 32, 1536),
    "large": ((), 8192, 64, 6144),
    "batched": ((8,), 1024, 16, 768),
}

STREAM_LENGTH = 8  # Distinct table shapes per table-stream measurement.

# Geometric row-bucket grid (~1.25-1.5x steps): padding waste is bounded
# by the step ratio while keeping the number of distinct shapes small.
# 10240 splits the 8192->12288 step: profiling measured 17% saved for
# tables landing in the lower half of that range.
ROW_BUCKETS = [
    16,
    24,
    32,
    48,
    64,
    96,
    128,
    192,
    256,
    384,
    512,
    768,
    1024,
    1536,
    2048,
    3072,
    4096,
    6144,
    8192,
    10240,
    12288,
    16384,
]


# Column buckets use ~12.5% steps: column padding costs compute roughly
# linearly, so steps are kept tighter than the row grid.
COL_BUCKETS = [
    4, 8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64, 72, 80, 96, 112, 128,
]  # fmt: skip


def bucket_rows(num_rows: int) -> int:
    """Round a positive row count up to the bucket grid (zero stays zero)."""
    if num_rows <= 0:
        return 0
    for size in ROW_BUCKETS:
        if num_rows <= size:
            return size
    return -(-num_rows // 4096) * 4096


def bucket_cols(num_cols: int) -> int:
    """Round a column count up to the bucket grid."""
    for size in COL_BUCKETS:
        if num_cols <= size:
            return size
    return -(-num_cols // 32) * 32


def pad_to_buckets(
    x: Tensor,
    y: Tensor,
) -> tuple[Tensor, Tensor, dict[str, Tensor], int]:
    """Pad train rows, test rows, and columns up to bucketed sizes.

    Padded train rows are masked from attention via ``seqused_train``,
    padded columns via ``seqused_cols``, and padded test rows produce
    extra predictions the caller slices away. Returns the padded ``x``,
    padded ``y``, the ``seqused_*`` keyword arguments, and the true
    test-row count. Tables whose train, test, and column counts already
    sit on the bucket grid return an *empty* ``seqused`` dict so they
    serve through the (faster) unmasked graph family; only off-grid
    tables carry ``seqused_*`` arguments.
    """
    num_train = y.size(-1)
    num_test = x.size(-2) - num_train
    num_cols = x.size(-1)
    padded_train = bucket_rows(num_train)
    padded_test = bucket_rows(num_test)
    padded_cols = bucket_cols(num_cols)
    batch_shape = x.shape[:-2]
    # Skip the copies when a dimension is already on its bucket boundary.
    if padded_cols != num_cols:
        x = torch.cat(
            [
                x,
                x.new_zeros(*batch_shape, x.size(-2), padded_cols - num_cols),
            ],
            dim=-1,
        )
    if padded_train != num_train or padded_test != num_test:
        x = torch.cat(
            [
                x[..., :num_train, :],
                x.new_zeros(
                    *batch_shape, padded_train - num_train, padded_cols
                ),
                x[..., num_train:, :],
                x.new_zeros(*batch_shape, padded_test - num_test, padded_cols),
            ],
            dim=-2,
        )
        # Pad the labels by repeating a real one instead of writing zeros.
        # Labels define the categorical category set, so a zero pad would
        # introduce a class the true labels never contain: it renumbers
        # every real row's code and widens the output by a column, which
        # is exactly what the padded-vs-unpadded gate compares.
        if num_train > 0:
            y = torch.cat(
                [
                    y,
                    y[..., :1].expand(*batch_shape, padded_train - num_train),
                ],
                dim=-1,
            )
        else:  # No real label to repeat; `expand` would fail on a 0-width y.
            y = y.new_zeros(*batch_shape, padded_train)
    # Exact-fit tables skip the seqused arguments entirely: masking costs
    # measurably more than unmasked attention (bool-mask SDPA), so the
    # unpadded graph family serves on-grid tables at full speed. Off-grid
    # tables take the masked family (warmed separately).
    seqused: dict[str, Tensor] = {}
    if (
        padded_train != num_train
        or padded_test != num_test
        or padded_cols != num_cols
    ):
        seqused = {
            "seqused_train": torch.full(
                batch_shape, num_train, dtype=torch.int32, device=x.device
            ),
            "seqused_cols": torch.tensor(
                num_cols, dtype=torch.int32, device=x.device
            ),
        }
    return x, y, seqused, num_test


def reachable_buckets(workload: str) -> set[tuple[int, int, int]]:
    """Enumerate the bucketed shapes a jittered workload stream can visit."""
    _, num_rows, num_columns, _ = WORKLOADS[workload]
    combos = set()
    for extra_rows in range(num_rows // 4):
        rows = num_rows + extra_rows
        train = (rows * 3) // 4
        for extra_cols in range(4):
            combos.add(
                (
                    bucket_rows(train),
                    bucket_rows(rows - train),
                    bucket_cols(num_columns + extra_cols),
                )
            )
    return combos


def make_table(
    workload: str,
    task: str,
    seed: int,
    device: torch.device,
    jitter: int = 0,
) -> tuple[Tensor, Tensor]:
    """Generate a seeded, signal-bearing table for the given workload."""
    batch_shape, num_rows, num_columns, num_train = WORKLOADS[workload]
    if jitter:
        generator = torch.Generator().manual_seed(seed * 1_000 + jitter)
        num_rows += int(
            torch.randint(0, num_rows // 4, (), generator=generator)
        )
        num_columns += int(torch.randint(0, 4, (), generator=generator))
        num_train = (num_rows * 3) // 4
    torch.manual_seed(seed)

    x = torch.randn(*batch_shape, num_rows, num_columns, device=device)
    if task == "cls":
        num_classes = 4
        labels = torch.randint(
            0, num_classes, (*batch_shape, num_rows), device=device
        )
        # Shift a subset of columns per class so logit margins are
        # meaningful (pure-noise tables make top-1 agreement a coin flip).
        shift = torch.randn(num_classes, num_columns, device=device)
        x = x + 2.0 * shift[labels]
        y = labels[..., :num_train]
    else:
        weights = torch.randn(num_columns, device=device)
        target = x @ weights + 0.1 * torch.randn(
            *batch_shape, num_rows, device=device
        )
        y = target[..., :num_train]
    return x, y


def apply_precision(model: TabICLv2, precision: str) -> None:
    """Apply the requested precision mode to globals and the model."""
    if precision == "fp32":
        matmul = torch.backends.cuda.matmul
        if hasattr(matmul, "fp32_precision"):  # torch>=2.9
            matmul.fp32_precision = "ieee"
        else:
            matmul.allow_tf32 = False
    elif precision in ("tf32", "bf16-autocast"):
        torch.set_float32_matmul_precision("high")
    elif precision == "bf16-full":
        torch.set_float32_matmul_precision("high")
        model.to(torch.bfloat16)
    elif precision == "fp16-full":
        torch.set_float32_matmul_precision("high")
        model.to(torch.float16)
    else:
        raise ValueError(f"unknown precision '{precision}'")


def cast_inputs(x: Tensor, y: Tensor, precision: str) -> tuple[Tensor, Tensor]:
    """Cast inputs to match the precision mode."""
    if precision in ("bf16-full", "fp16-full"):
        dtype = torch.bfloat16 if precision == "bf16-full" else torch.float16
        x = x.to(dtype)
        if y.is_floating_point():
            y = y.to(dtype)
    return x, y


def load_transformer_engine() -> tuple[Any, Any]:
    """Import Transformer Engine lazily (optional heavy dependency)."""
    te = importlib.import_module("transformer_engine.pytorch")
    recipes = importlib.import_module("transformer_engine.common.recipe")
    return te, recipes


def make_te_recipe(recipes: Any, name: str) -> Any:
    """Build the Transformer Engine scaling recipe for ``name``."""
    if name == "fp8":
        return recipes.DelayedScaling()
    if name == "mxfp8":
        return recipes.MXFP8BlockScaling()
    if name == "nvfp4":
        return recipes.NVFP4BlockScaling()
    raise ValueError(f"unknown recipe '{name}'")


def swap_te_linears(model: TabICLv2, te: Any) -> int:
    """Swap eligible ``Linear`` modules for ``te.Linear``.

    Transformer Engine low-precision GEMMs require both feature
    dimensions to be multiples of 16 and the product of the leading
    dimensions to be divisible by 8. Ineligible layers stay in bf16:
    the feature-grouping projection and output head (feature dims), and
    the :class:`~sdm.nn.QASSMax` scaling MLPs (which can see a single
    key-length row, so their leading product can be 1).
    """
    excluded: set[int] = set()
    for module in model.modules():
        if isinstance(module, QASSMax):
            excluded.update(id(m) for m in module.modules())

    swapped = 0
    for module in model.modules():
        if id(module) in excluded:
            continue
        for name, child in list(module.named_children()):
            if (
                isinstance(child, torch.nn.Linear)
                and child.in_features % 16 == 0
                and child.out_features % 16 == 0
            ):
                replacement = te.Linear(
                    child.in_features,
                    child.out_features,
                    bias=child.bias is not None,
                    params_dtype=child.weight.dtype,
                    device=child.weight.device,
                )
                with torch.no_grad():
                    replacement.weight.copy_(child.weight)
                    if child.bias is not None:
                        replacement.bias.copy_(child.bias)
                setattr(module, name, replacement)
                swapped += 1
    return swapped


def cudnn_attention_cell(config: str) -> bool:
    """Whether a configuration's attention can run on the cuDNN SDPA backend.

    The backend takes fp16/bf16 only; torch's default order and the
    cuDNN-first pin select it, a pinned order without it does not.
    """
    precision, sdpa_priority, *_ = CONFIGS[config]
    if precision in ("fp32", "tf32"):
        return False
    if isinstance(sdpa_priority, bool):
        return True
    return "CUDNN_ATTENTION" in sdpa_priority


def bucketed_cell(config: str) -> bool:
    """Whether a configuration pads its tables to the bucket grid."""
    extras = CONFIGS[config][4:]
    return bool(extras and extras[0])


def varlen_engaged(built: int, failed: int, exercised: bool) -> bool:
    """Whether numbers may be published under the variable-length label.

    ``built``/``failed`` are the :func:`sdm.nn.cudnn_varlen_stats` counts
    after the cell ran and ``exercised`` records whether any call passed
    ``seqused`` counts: no shape may have degraded to the masked fallback,
    and once counts were passed at least one execution graph must exist.
    A cell that never passed counts is legitimately idle.
    """
    return not failed and (built > 0 or not exercised)


def cuda_cache_file_count() -> int:
    """Count the files in the CUDA driver's JIT cache directory.

    The cuDNN attention plan build JIT-compiles its kernel through the
    driver, which stores the result here per kernel variant (the small- and
    large-table attention kernels are distinct compiles, so a directory
    warmed by one table-size class still grows for the other), but the
    directory also receives torch's own runtime-JIT'd kernels on first use
    (an eager fp32 cell adds a few files to an empty directory with a flat
    stream), so growth is attributable to the plan build only for cells
    that reach cuDNN attention - and to inflated fresh-table stream numbers
    only for the unbucketed ones, since bucket warmup builds every plan
    before the stream starts.
    """
    path = os.environ.get("CUDA_CACHE_PATH") or os.path.expanduser(
        "~/.nv/ComputeCache"
    )
    return sum(len(files) for _, _, files in os.walk(path))


def timed_loop(fn: Any, min_seconds: float = 3.0) -> dict[str, float]:
    """Measure sync-bounded wall-clock latency of ``fn``."""
    times: list[float] = []
    while sum(times) < min_seconds or len(times) < 30:
        if len(times) >= 1000:
            break
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    times.sort()
    n = len(times)
    return {
        "p50_s": statistics.median(times),
        "iqr_s": times[(3 * n) // 4] - times[n // 4],
        "min_s": times[0],
        "iters": n,
    }


def accuracy_block(
    task: str,
    out: Tensor,
    ref: Tensor,
) -> dict[str, float | bool]:
    """Compare a configuration's output against the fp32 reference."""
    out = out.float()
    ref = ref.float()
    block: dict[str, float | bool] = {
        "max_abs_diff": (out - ref).abs().max().item()
    }
    if task == "cls":
        agree = (out.argmax(-1) == ref.argmax(-1)).float().mean().item()
        top2 = ref.topk(2, dim=-1).values
        # Per-row worst logit shift relative to that row's own decision
        # margin: a flip-risk statistic rather than a global average.
        margin = (top2[..., 0] - top2[..., 1]).clamp(min=1e-9)
        delta = (out - ref).abs().amax(-1)
        block["top1_agreement"] = agree
        block["margin_ratio"] = (delta / margin).median().item()
        block["pass"] = agree >= 0.995 and block["margin_ratio"] < 0.1
    else:
        # Scale errors by each row's predicted quantile span (robust to
        # near-zero central quantiles and independent of quantile count).
        span = (ref.amax(-1) - ref.amin(-1)).clamp(min=1e-9)
        rel = (out - ref).abs() / span.unsqueeze(-1)
        block["median_rel_err"] = rel.median().item()
        block["p99_rel_err"] = rel.flatten().quantile(0.99).item()
        block["pass"] = (
            block["median_rel_err"] < 1e-2 and block["p99_rel_err"] < 5e-2
        )
    return block


def run_cell(spec: dict[str, Any], workdir: str) -> dict[str, Any]:
    """Run one (config, workload, task, route) benchmark cell."""
    config, workload = spec["config"], spec["workload"]
    task, route = spec["task"], spec["route"]
    precision, sdpa_priority, compile_kwargs, te_recipe, *extras = CONFIGS[
        config
    ]
    bucketed = bool(extras and extras[0])
    device = torch.device("cuda")
    result: dict[str, Any] = dict(spec)
    cuda_cache_files_before = cuda_cache_file_count()
    # Whether this cell ever passed a non-empty seqused_* set, i.e. actually
    # requested the variable-length path. Exact-fit tables skip seqused by
    # design (masking costs), so a cell can be legitimately varlen-idle.
    varlen_exercised = [False]
    if len(extras) > 1 and extras[1]:
        result["cudnn_varlen_active"] = enable_cudnn_varlen(True)
        if not result["cudnn_varlen_active"]:
            raise RuntimeError(
                f"config '{config}' requests the cuDNN variable-length "
                f"path but it is inert (nvidia-cudnn-frontend missing or "
                f"unimportable); refusing to publish boolean-mask numbers "
                f"under the variable-length label"
            )

    model = TabICLv2(pretrained=True, device=device)
    apply_precision(model, precision)

    # Context factories: `sdpa_kernel` is a generator-based context manager
    # and therefore single-use, so build a fresh instance per call.
    context_factories: list[Any] = []
    backends = sdpa_backends(sdpa_priority)
    # The backend order decides whether the cell pays the cuDNN
    # fresh-shape tax, so the record names it.
    result["sdpa_priority"] = (
        None if backends is None else [backend.name for backend in backends]
    )
    if backends is not None:
        pinned = backends
        context_factories.append(
            lambda: sdpa_kernel(pinned, set_priority=True)
        )
    if precision == "bf16-autocast":
        context_factories.append(
            lambda: torch.amp.autocast("cuda", torch.bfloat16)
        )
    if te_recipe is not None:
        te, recipes = load_transformer_engine()
        result["te_swapped_linears"] = swap_te_linears(model, te)
        fp8_recipe = make_te_recipe(recipes, te_recipe)
        context_factories.append(
            lambda: te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe)
        )

    # Every record names the device and software stack it was measured on,
    # so numbers lifted from the JSON stay attributable.
    result["environment"] = {
        "device": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        # The cuDNN attention plan build JIT-compiles its kernel through
        # the driver, which caches the result on disk per kernel variant;
        # a fresh-shape cost is a warm- or cold-cache number depending on
        # whether the directory already holds this cell's table-size class
        # (from any earlier process or container session), and
        # ``cuda_cache_files_added`` (filled in at the end) says whether
        # this cell hit the driver compiler. The count is raw: torch's own
        # JIT'd kernels land in the same directory, so it marks a
        # plan-build miss only for cells that reach cuDNN attention.
        "cuda_cache_disabled": os.environ.get("CUDA_CACHE_DISABLE") == "1",
        "cuda_cache_path": os.environ.get("CUDA_CACHE_PATH"),
        "cudnn": torch.backends.cudnn.version(),
        "sdm": sdm.__version__,
    }
    if result.get("cudnn_varlen_active"):
        # The variable-length path runs cudnn-frontend execution graphs,
        # so its numbers are attributable to that package version too.
        result["environment"]["cudnn_frontend"] = importlib.import_module(
            "cudnn"
        ).__version__

    def call(x: Tensor, y: Tensor, **kwargs: Any) -> Tensor:
        # `x` holds the in-context rows followed by the query rows; `y`
        # holds the in-context targets.
        num_train = y.size(-1)
        with ExitStack() as stack:
            for factory in context_factories:
                stack.enter_context(factory())
            out = model(
                x[..., :num_train, :],
                y.unsqueeze(-1),
                x[..., num_train:, :],
                recipe=Recipe(),
                num_estimators=1,
                **kwargs,
            )
        # A pass-through recipe keeps the ensemble dimension, and the
        # padded cells require it: fitted pre-processing would derive its
        # state from the padded rows, violating the `seqused_*` contract.
        # The explicit `num_estimators=1` keeps the batch interpretation
        # of higher-rank inputs: with `num_estimators=None`, the `batched`
        # workload's leading dimension would be consumed as the estimator
        # dimension, so `out.numerical[0]` would return one batch element
        # instead of the full batch, and the padded cells would be
        # rejected as ambiguous by the `seqused_*` validation.
        return out.numerical[0]

    def bucketed_call(x: Tensor, y: Tensor) -> Tensor:
        """Pad to bucketed shapes, run, and slice the true test rows."""
        x, y, seqused, num_test = pad_to_buckets(x, y)
        if seqused:
            varlen_exercised[0] = True
        out = call(x, y, **seqused)
        return out[..., :num_test, :]

    stream_call = bucketed_call if bucketed else call

    if compile_kwargs is not None:
        compile_kwargs = dict(compile_kwargs)
        if compile_kwargs.pop("regional", False):
            if compile_kwargs.get("mode") == "reduce-overhead":
                raise ValueError(
                    "regional compilation is supported with the default "
                    "compile mode only: the first ICL layer hands a "
                    "cudagraph-pool tensor to the later blocks' `out=` "
                    "buffers, which CUDA graph trees reject"
                )
            # All block instances share one ``TransformerBlock.forward``
            # code object, and the fit/predict route adds KV-cache record
            # and replay specializations on top of the per-site shapes, so
            # the route exceeds Dynamo's default ``recompile_limit`` of 8.
            # With ``fullgraph=True`` that raises
            # ``FailOnRecompileLimitHit`` instead of falling back to eager.
            torch._dynamo.config.recompile_limit = 64  # ty: ignore[invalid-assignment]
            for module in model.modules():
                if isinstance(
                    module, (TransformerBlock, InducedTransformerBlock)
                ):
                    module.compile(**compile_kwargs)
            # The prediction heads stay eager: each is a bare
            # ``nn.Sequential`` of torch built-ins whose forwards Dynamo
            # skips, so a ``fullgraph=True`` compile finds no frames to
            # compile - a silent no-op through torch 2.12 and a
            # RuntimeError at the first call on torch 2.13.
        else:
            for sub_model in model.models.values():
                sub_model.compile(**compile_kwargs)

    x, y = make_table(workload, task, seed=0, device=device)
    x, y = cast_inputs(x, y, precision)

    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    out = stream_call(x, y).clone()
    torch.cuda.synchronize()
    result["cold_first_call_s"] = time.perf_counter() - start

    if route == "oneshot":
        for _ in range(3):  # Warmup (post-compile).
            stream_call(x, y)
        result.update(timed_loop(lambda: stream_call(x, y)))

        # Table-stream: fresh shapes, including any recompiles. Jittered
        # shapes can violate low-precision GEMM divisibility constraints;
        # such failures are counted rather than aborting the cell. A second
        # pass over NEW jittered tables separates one-time shape warmup
        # (bucketed configs revisit warm buckets; unbucketed configs keep
        # paying per fresh shape) from the steady stream cost.
        stream_errors = 0
        for stream_pass, tables_key, jitters in (
            (
                "table_stream_per_table_s",
                "table_stream_tables_s",
                range(1, STREAM_LENGTH + 1),
            ),
            (
                "table_stream_fresh2_per_table_s",
                "table_stream_fresh2_tables_s",
                range(STREAM_LENGTH + 1, 2 * STREAM_LENGTH + 1),
            ),
        ):
            if bucketed and stream_pass == "table_stream_per_table_s":
                # Boot-style warmup: visit every bucket combination the
                # jittered stream can reach so the stream passes measure
                # the warm serving regime (the first pass still starts with
                # one table slightly above the rest; the per-table times keep
                # it visible). The serving cold cost is the
                # (separately reported) cold_first_call_s plus this
                # bucket_warmup_s.
                warm_start = time.perf_counter()
                batch_shape = WORKLOADS[workload][0]
                for train_b, test_b, cols_b in sorted(
                    reachable_buckets(workload)
                ):
                    xw = torch.zeros(
                        *batch_shape, train_b + test_b, cols_b, device=device
                    )
                    if task == "cls":
                        yw = torch.zeros(
                            *batch_shape,
                            train_b,
                            dtype=torch.long,
                            device=device,
                        )
                    else:
                        yw = torch.zeros(*batch_shape, train_b, device=device)
                    xw, yw = cast_inputs(xw, yw, precision)
                    varlen_exercised[0] = True
                    for _ in range(2):  # Graph capture needs a re-visit.
                        # Only the masked family is warmed here. The unmasked
                        # family serves exactly-on-grid tables, which for
                        # every shipped workload is only the base shape - and
                        # that is already warmed by the cold call plus the
                        # post-compile warmups above.
                        call(
                            xw,
                            yw,
                            seqused_train=torch.full(
                                batch_shape,
                                train_b,
                                dtype=torch.int32,
                                device=device,
                            ),
                            seqused_cols=torch.tensor(
                                cols_b, dtype=torch.int32, device=device
                            ),
                        )
                torch.cuda.synchronize()
                result["bucket_warmup_s"] = time.perf_counter() - warm_start
            # The GPU idles down to its lowest clocks during the CPU-bound
            # compile and bucket warmup above; a few untimed calls on the
            # already-warm base shape bring it back to serving clocks so
            # the pass measures the warm serving regime rather than clock
            # ramp-up on its first tables.
            for _ in range(3):
                stream_call(x, y)
            torch.cuda.synchronize()
            # Per-table times (each bounded by a synchronize) keep the
            # dispersion within a pass; the headline is their mean. The
            # clock starts once the table exists: generating it is a
            # benchmark artifact (~0.2 ms on GB200, a few percent of the
            # smallest bucketed stream cost), while the padding inside
            # ``bucketed_call`` is part of the serving path and stays in.
            table_times: list[float] = []
            for jitter in jitters:
                xs, ys = make_table(
                    workload, task, seed=jitter, device=device, jitter=jitter
                )
                xs, ys = cast_inputs(xs, ys, precision)
                torch.cuda.synchronize()
                table_start = time.perf_counter()
                try:
                    stream_call(xs, ys)
                except (RuntimeError, ValueError):
                    # Low-precision GEMMs constrain shapes (for example fp8
                    # requires leading-dimension products divisible by 8), so
                    # arbitrary table sizes may need padding.
                    stream_errors += 1
                torch.cuda.synchronize()
                table_times.append(time.perf_counter() - table_start)
            result[stream_pass] = sum(table_times) / STREAM_LENGTH
            result[tables_key] = table_times
        if stream_errors:
            result["table_stream_errors"] = stream_errors
            result["table_stream_per_table_s"] = None
            result["table_stream_fresh2_per_table_s"] = None
        # Re-measure the on-grid latency after the stream passes: a stream
        # timed during a transient slowdown (a neighbour on the GPU, clock
        # throttling) then shows up as drift in the record instead of being
        # published silently. Mirrors the driver-level c0 sentinel.
        result["stream_drift"] = (
            timed_loop(lambda: stream_call(x, y))["p50_s"] / result["p50_s"]
        )
    else:  # fit/predict route.
        num_train = WORKLOADS[workload][3]
        x_train, x_test = x[..., :num_train, :], x[..., num_train:, :]

        fit_kwargs: dict[str, Any] = {}
        if bucketed:
            x_padded, y, seqused, _ = pad_to_buckets(x_train, y)
            x_train = x_padded[..., : y.size(-1), :]
            fit_kwargs.update(seqused)
            if seqused:
                varlen_exercised[0] = True

        uses_cudagraphs = compile_kwargs is not None and "reduce-overhead" in (
            str(compile_kwargs.get("mode", ""))
        )

        def clone_cached_kv() -> None:
            # Under reduce-overhead, the key/value projections recorded by
            # fit are CUDA-graph-pool outputs that later replays overwrite.
            # Give the cache its own storage once per fit. (Accesses cache
            # internals: the serving-side pattern pending a public API.)
            cache = model._cache
            if not uses_cudagraphs or cache is None:
                return
            num_estimators = cast(
                RecipeExecution, cache["recipe_execution"]
            ).num_members
            for i in range(num_estimators):
                items = cast(Cache, cache[i])._items
                for cache_key, value in list(items.items()):
                    if isinstance(value, KVCacheEntry):
                        items[cache_key] = KVCacheEntry(
                            key=value.key.clone(), value=value.value.clone()
                        )

        def run_predict(x_test: Tensor) -> Tensor:
            if not bucketed:
                return model.predict(x_test).numerical[0]
            num_test = x_test.size(-2)
            padded_test = bucket_rows(num_test)
            # Match the fitted column padding. Skip the copies when a
            # dimension is already on its bucket boundary (as
            # `pad_to_buckets` does) so the timed loop does not pay for a
            # zero-width concatenation.
            padded_cols = x_train.size(-1)
            if padded_cols != x_test.size(-1):
                x_test = torch.cat(
                    [
                        x_test,
                        x_test.new_zeros(
                            *x_test.shape[:-2],
                            num_test,
                            padded_cols - x_test.size(-1),
                        ),
                    ],
                    dim=-1,
                )
            if padded_test != num_test:
                x_test = torch.cat(
                    [
                        x_test,
                        x_test.new_zeros(
                            *x_test.shape[:-2],
                            padded_test - num_test,
                            padded_cols,
                        ),
                    ],
                    dim=-2,
                )
            return model.predict(x_test).numerical[0][..., :num_test, :]

        def fit_predict() -> Tensor:
            with ExitStack() as stack:
                for factory in context_factories:
                    stack.enter_context(factory())
                model.fit(
                    x_train,
                    y.unsqueeze(-1),
                    recipe=Recipe(),
                    **fit_kwargs,
                )
                clone_cached_kv()
                pred = run_predict(x_test)
                model.clear()
                return pred

        def predict_only() -> Tensor:
            with ExitStack() as stack:
                for factory in context_factories:
                    stack.enter_context(factory())
                return run_predict(x_test)

        # The one-shot cold call above runs the whole table in one pass and
        # peaks ~30% above fit/predict; reset so `peak_mem_mb` describes
        # this route rather than that call.
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        out = fit_predict().clone()
        torch.cuda.synchronize()
        # cold_first_call_s covers the one-shot family; the fit/predict
        # route compiles its own record/replay graphs on this first call.
        result["fitpredict_cold_s"] = time.perf_counter() - start
        fit_predict()  # Warmup both graphs.
        torch.cuda.synchronize()
        with ExitStack() as stack:
            for factory in context_factories:
                stack.enter_context(factory())
            start = time.perf_counter()
            model.fit(
                x_train,
                y.unsqueeze(-1),
                recipe=Recipe(),
                **fit_kwargs,
            )
            # The clone is a mandatory part of fitting under CUDA graphs,
            # so it belongs inside the timed region.
            clone_cached_kv()
            torch.cuda.synchronize()
            result["fit_s"] = time.perf_counter() - start
        predict_only()
        result.update(timed_loop(predict_only))
        model.clear()

    result["peak_mem_mb"] = torch.cuda.max_memory_allocated() / 2**20

    if route == "oneshot":
        # Pool the accuracy check over several seeded tables so gates do
        # not hinge on a handful of test rows from a single table. Bucketed
        # configs run the same call here (the canonical shapes sit on the
        # bucket grid, so no padding is exercised); padding itself is gated
        # below on an off-grid table.
        pooled = []
        for seed in range(5):
            xs, ys = make_table(workload, task, seed=seed, device=device)
            xs, ys = cast_inputs(xs, ys, precision)
            pooled.append(stream_call(xs, ys).clone().float().cpu())
        out = torch.stack(pooled)

    if route == "oneshot" and bucketed:
        # The pooled seeds use the canonical workload shapes, which land
        # exactly on the bucket grids - no padding is exercised. Gate the
        # padding itself on an off-grid jittered table against the same
        # configuration's unpadded call (same precision, so the comparison
        # isolates padding correctness with tight thresholds).
        xs, ys = make_table(workload, task, seed=97, device=device, jitter=97)
        xs, ys = cast_inputs(xs, ys, precision)
        direct = call(xs, ys).clone().float().cpu()
        padded = stream_call(xs, ys).clone().float().cpu()
        gate = accuracy_block(task, padded, direct)
        gate["padded_rows"] = bucket_rows(ys.size(-1)) - ys.size(-1)
        gate["padded_cols"] = bucket_cols(xs.size(-1)) - xs.size(-1)
        if task == "cls":
            gate["pass"] = (
                gate["top1_agreement"] >= 0.995 and gate["margin_ratio"] < 0.05
            )
        else:
            # Floors calibrated from the first GB200 measurement of
            # same-precision padded-vs-unpadded noise (c18/small reg,
            # median 3.2e-3 / p99 1.3e-2): compiled padded and unpadded
            # shapes autotune different kernels, so bf16 divergence sits
            # near the casting-error scale even at same precision. ~3x
            # headroom over the measurement; real masking leakage measures
            # >= 1.7e-1 on the leak-sensitivity tests, two orders louder.
            gate["pass"] = (
                gate["median_rel_err"] < 1e-2 and gate["p99_rel_err"] < 5e-2
            )
        result["padding_gate"] = gate

    # Accuracy vs the fp32 reference produced by the c0 cell.
    ref_path = os.path.join(workdir, f"ref-{workload}-{task}-{route}.pt")
    if config == "c0-fp32" and not spec.get("sentinel"):
        torch.save(out.float().cpu(), ref_path)
        result["accuracy"] = {"pass": True, "is_reference": True}
    elif config == "c0-fp32":
        result["accuracy"] = {"pass": True, "is_sentinel": True}
        if spec.get("reference_p50_s"):
            result["sentinel_drift"] = (
                result["p50_s"] / spec["reference_p50_s"]
            )
    elif os.path.exists(ref_path):
        ref = torch.load(ref_path, map_location="cpu")
        result["accuracy"] = accuracy_block(task, out.float().cpu(), ref)
    else:
        result["accuracy"] = {"pass": None, "missing_reference": True}

    if result.get("cudnn_varlen_active"):
        # Import availability alone does not prove the kernel served the
        # cell: an unsupported shape degrades to the masked fallback
        # inside the op (probed once per shape, negatively cached, warned
        # once), and the warning is lost on green runs because the parent
        # keeps subprocess stderr only on failure.
        built, failed = cudnn_varlen_stats()
        result["cudnn_varlen_graphs_built"] = built
        result["cudnn_varlen_build_failures"] = failed
        result["cudnn_varlen_exercised"] = varlen_exercised[0]
        # Exact-fit cells (canonical shapes land on the bucket grid) never
        # pass seqused, so the varlen path is legitimately idle and the
        # numbers are mask-free by construction; demanding engagement there
        # is a false alarm. The gate fires when varlen was REQUESTED but a
        # shape degraded (failed > 0) or nothing engaged. For this model
        # and these workloads every dispatch-level eligibility term is
        # cell-constant (dtype, heads, head dim, and the eager chunking
        # cap on the flattened batch), so eligibility rejection is
        # all-or-nothing per cell and lands in the nothing-engaged
        # branch; per-call op-level metadata fallbacks are unreachable
        # from TabICLv2's call sites and are not counted here.
        if not varlen_engaged(built, failed, varlen_exercised[0]):
            raise RuntimeError(
                f"config '{config}' requests the cuDNN variable-length "
                f"path but it did not fully engage ({built} execution "
                f"graph(s) built, {failed} shape(s) degraded to the "
                f"masked fallback); refusing to publish boolean-mask "
                f"numbers under the variable-length label"
            )
    result["environment"]["cuda_cache_files_added"] = (
        cuda_cache_file_count() - cuda_cache_files_before
    )
    return result


def spawn_cell(
    spec: dict[str, Any],
    out_path: str,
    workdir: str,
) -> dict[str, Any] | None:
    """Run one cell in an isolated subprocess and return its record."""
    env = dict(os.environ)
    env["TORCHINDUCTOR_CACHE_DIR"] = tempfile.mkdtemp(
        prefix=f"inductor-{spec['config']}-", dir=workdir
    )
    try:
        proc = subprocess.run(
            [
                sys.executable,
                os.path.abspath(__file__),
                "--run-cell",
                json.dumps(spec),
                "--out",
                out_path,
                "--workdir",
                workdir,
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=1800,
        )
    except subprocess.TimeoutExpired as exc:
        # Record a timed-out cell like any other failed cell so the driver
        # moves on to the next one. `TimeoutExpired.stderr` holds raw bytes
        # (or None) even under text=True, so decode before serializing.
        stderr = exc.stderr or b""
        if isinstance(stderr, bytes):
            stderr = stderr.decode(errors="replace")
        record = dict(spec)
        record["error"] = (
            f"cell timeout after {exc.timeout}s: {stderr[-2000:]}"
        )
        with open(out_path, "a") as handle:
            handle.write(json.dumps(record) + "\n")
        return record
    if proc.returncode != 0:
        record = dict(spec)
        record["error"] = proc.stderr[-2000:]
        with open(out_path, "a") as handle:
            handle.write(json.dumps(record) + "\n")
        return record
    with open(out_path) as handle:
        return json.loads(handle.readlines()[-1])


def main() -> None:
    """Plan and execute the benchmark grid."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="bench.json")
    parser.add_argument("--budget-minutes", type=float, default=110.0)
    parser.add_argument(
        "--configs",
        default=",".join(CONFIGS),
        help="Comma-separated subset of configurations to benchmark "
        f"(default: all). Choices: {', '.join(CONFIGS)}.",
    )
    parser.add_argument("--run-cell", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--workdir", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    # Every cell requires CUDA (run_cell targets torch.device("cuda") and
    # the timing loops synchronize the device); fail once and clearly
    # instead of spawning a grid of subprocesses that each crash inside
    # model construction.
    if not torch.cuda.is_available():
        raise SystemExit("benchmark_tabiclv2.py requires a CUDA device")

    if args.run_cell is not None:
        record = run_cell(json.loads(args.run_cell), args.workdir)
        with open(args.out, "a") as handle:
            handle.write(json.dumps(record) + "\n")
        return

    workdir = tempfile.mkdtemp(prefix="sdm-bench-")
    deadline = time.time() + args.budget_minutes * 60.0
    records: list[dict[str, Any]] = []

    def enqueue(cells: list[dict[str, Any]]) -> None:
        for spec in cells:
            # Sentinels are cheap and bracket whatever did run, so the
            # walltime budget never drops them.
            if time.time() > deadline and not spec.get("sentinel"):
                spec = dict(spec)
                spec["skipped_walltime"] = True
                records.append(spec)
                with open(args.out, "a") as handle:
                    handle.write(json.dumps(spec) + "\n")
                continue
            record = spawn_cell(spec, args.out, workdir)
            if record is not None:
                records.append(record)
                print(format_record(record), flush=True)

    try:
        selected = [c.strip() for c in args.configs.split(",") if c.strip()]
        unknown = [c for c in selected if c not in CONFIGS]
        if unknown:
            raise SystemExit(
                f"unknown --configs entries: {', '.join(unknown)}"
            )
        # The accuracy reference must exist before any dependent cell runs.
        selected = ["c0-fp32"] + [c for c in selected if c != "c0-fp32"]

        # Stage 1: full config grid on the size extremes (classification,
        # one-shot route). c0 first: it writes the accuracy reference.
        stage1 = [
            {
                "config": config,
                "workload": workload,
                "task": "cls",
                "route": "oneshot",
            }
            for workload in ("small", "large")
            for config in selected
        ]
        enqueue(stage1)

        # Sentinel: re-measure the baseline to detect clock/thermal drift
        # (the record carries the p50 ratio against the reference cell).
        reference_p50_s = next(
            (
                r["p50_s"]
                for r in records
                if r["config"] == "c0-fp32"
                and r["workload"] == "large"
                and r.get("p50_s")
            ),
            None,
        )
        sentinel = {
            "config": "c0-fp32",
            "workload": "large",
            "task": "cls",
            "route": "oneshot",
            "sentinel": True,
            "reference_p50_s": reference_p50_s,
        }
        enqueue([sentinel])

        # Stage 2: promote the top-2 non-baseline configs by large-table
        # latency to the remaining workloads, tasks, and the fit/predict route.
        scored = [
            r
            for r in records
            if r.get("workload") == "large"
            and r.get("p50_s")
            and r["config"] != "c0-fp32"
            and r.get("accuracy", {}).get("pass")
            # A cell whose off-grid padding gate failed has invalid bucketed
            # numbers, and a cell that errored on jittered stream tables has
            # no honest per-fresh-table cost: neither may promote.
            and r.get("padding_gate", {"pass": True}).get("pass")
            and not r.get("table_stream_errors")
        ]
        # Rank by the worst of steady-state and both per-fresh-table passes so
        # a config that recompiles per shape cannot win on p50 alone, and keep
        # the example recipe in the promoted set.
        top = [
            r["config"]
            for r in sorted(
                scored,
                key=lambda r: (
                    max(
                        r["p50_s"],
                        r.get("table_stream_per_table_s") or r["p50_s"],
                        r.get("table_stream_fresh2_per_table_s") or r["p50_s"],
                    ),
                    r["p50_s"],
                ),
            )[:2]
        ]
        if "c12-autocast-compile" in selected:
            top = list(dict.fromkeys([*top, "c12-autocast-compile"]))
        stage2 = []
        for config in ["c0-fp32", *top]:
            for workload in ("medium", "batched"):
                stage2.append(
                    {
                        "config": config,
                        "workload": workload,
                        "task": "cls",
                        "route": "oneshot",
                    }
                )
            for workload in ("small", "large"):
                stage2.append(
                    {
                        "config": config,
                        "workload": workload,
                        "task": "reg",
                        "route": "oneshot",
                    }
                )
                stage2.append(
                    {
                        "config": config,
                        "workload": workload,
                        "task": "cls",
                        "route": "fitpredict",
                    }
                )
        enqueue(stage2)

        # Second sentinel: stage 2 cells sit between the two sentinel rows,
        # so a slowdown that started after the first one is visible in the
        # summary instead of only in cells that happen to look off.
        enqueue([sentinel])

        print("\n=== SUMMARY (per-table stream / steady-state p50) ===")
        for record in records:
            print(format_record(record))
    finally:
        # The workdir holds the fp32 accuracy references plus one Inductor
        # cache per spawned cell; a full run would otherwise strand
        # hundreds of MB in the system temp directory. Keep it only when
        # a cell errored, so the caches stay inspectable.
        if any("error" in record for record in records):
            print(f"workdir kept for debugging: {workdir}", flush=True)
        else:
            shutil.rmtree(workdir, ignore_errors=True)


# Width of the config column in the summary table, sized to the longest
# configuration name so every row variant stays aligned.
_CONFIG_WIDTH = max(len(name) for name in CONFIGS)


def format_record(record: dict[str, Any]) -> str:
    """Format one benchmark record as a fixed-width table row."""
    if record.get("skipped_walltime"):
        return "{:<{w}} {:<8} {:<4} {:<10} SKIPPED (walltime)".format(
            record["config"],
            record["workload"],
            record["task"],
            record["route"],
            w=_CONFIG_WIDTH,
        )
    if "error" in record:
        return "{:<{w}} {:<8} {:<4} {:<10} ERROR".format(
            record["config"],
            record["workload"],
            record["task"],
            record["route"],
            w=_CONFIG_WIDTH,
        )
    accuracy = record.get("accuracy", {})
    # Fold the off-grid padding gate into the headline pass: a bucketed
    # cell whose padded-vs-unpadded gate failed must not print pass=True
    # (the full gate block stays in the JSON record).
    cell_pass = accuracy.get("pass")
    if not record.get("padding_gate", {"pass": True}).get("pass"):
        cell_pass = False
    # Stream errors record `None` for the per-table time; keep the driver
    # printing instead of crashing on the multiplication below.
    stream_s = record.get("table_stream_per_table_s")
    if stream_s is None:
        stream_s = float("nan")
    stream2_s = record.get("table_stream_fresh2_per_table_s")
    if stream2_s is None:
        stream2_s = float("nan")
    if record.get("route") == "fitpredict":
        # p50 is the cached-predict latency; the fit and the route's own
        # record/replay compile cost are separate fields, so print them
        # instead of the one-shot family's stream and cold columns.
        row = (
            "{:<{w}} {:<8} {:<4} {:<10} p50={:>8.2f}ms fit={:>8.2f}ms "
            "cold={:>6.1f}s mem={:>7.0f}MB pass={}".format(
                record["config"],
                record["workload"],
                record["task"],
                record["route"],
                1000 * record.get("p50_s", float("nan")),
                1000 * record.get("fit_s", float("nan")),
                record.get("fitpredict_cold_s", float("nan")),
                record.get("peak_mem_mb", float("nan")),
                cell_pass,
                w=_CONFIG_WIDTH,
            )
        )
    else:
        row = (
            "{:<{w}} {:<8} {:<4} {:<10} p50={:>8.2f}ms stream={:>8.2f}ms "
            "stream2={:>8.2f}ms cold={:>6.1f}s mem={:>7.0f}MB pass={}".format(
                record["config"],
                record["workload"],
                record["task"],
                record["route"],
                1000 * record.get("p50_s", float("nan")),
                1000 * stream_s,
                1000 * stream2_s,
                record.get("cold_first_call_s", float("nan")),
                record.get("peak_mem_mb", float("nan")),
                cell_pass,
                w=_CONFIG_WIDTH,
            )
        )
        # Bucketed cells pay a second one-time cost after the cold call
        # (the masked graph family plus every reachable bucket shape); the
        # serving cold start is the sum, so print both.
        if record.get("bucket_warmup_s") is not None:
            row += f" warm={record['bucket_warmup_s']:>5.1f}s"
        # A first table far above the rest of its pass carried a one-time
        # JIT cost (the driver compiling a cuDNN attention kernel on a CUDA
        # cache miss, or a Dynamo recompile) that inflates the stream=
        # column; stream2= is the steady value. A cuDNN-attention cell that
        # grew the CUDA cache, or ran with it disabled, compiled on every
        # miss - on a cold cache every fresh table pays alike, so the
        # first-table ratio alone would not notice. The file count is raw
        # (torch's own JIT'd kernels land in the same directory, even for
        # the eager fp32 cells), so the cache state counts only for cells
        # whose attention can reach that backend, and only for unbucketed
        # ones: bucket warmup builds every reachable plan before the timed
        # stream, so a bucketed cell's misses land in warm=, not stream=.
        # Say so either way.
        tables = record.get("table_stream_tables_s") or []
        environment = record.get("environment", {})
        config = record["config"]
        cache_miss = (
            (
                environment.get("cuda_cache_files_added", 0) > 0
                or environment.get("cuda_cache_disabled")
            )
            and cudnn_attention_cell(config)
            and not bucketed_cell(config)
        )
        if cache_miss or (
            len(tables) > 1 and tables[0] > 3 * statistics.median(tables)
        ):
            row += " JIT"
    # A cell whose own interquartile range exceeds 10% of its p50 was timed
    # in a noisy window; say so next to the number.
    p50_s = record.get("p50_s")
    iqr_s = record.get("iqr_s")
    if p50_s and iqr_s is not None and iqr_s / p50_s > 0.1:
        row += " NOISY"
    # The sentinel re-measures the fp32 reference after each stage and
    # every one-shot cell re-measures its own on-grid p50 after the stream
    # passes; a ratio off by more than 5% flags clock/thermal drift.
    for key, label in (
        ("sentinel_drift", "drift"),
        ("stream_drift", "sdrift"),
    ):
        drift = record.get(key)
        if drift is not None:
            row += f" {label}={drift:.3f}x"
            if abs(drift - 1.0) > 0.05:
                row += " DRIFT"
    return row


if __name__ == "__main__":
    main()
