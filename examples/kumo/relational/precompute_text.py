r"""Precompute frozen Qwen embeddings for the RelArena model/system examples.

Run from the repository root after the setup in README.relarena.md::

    python -m examples.kumo.relational.precompute_text \
        --dataset rel-amazon --task user-ltv --device cuda:0 \
        --output-dir /data/relarena/amazon-user-ltv --batch-size 1024

Pass the same directory as the runner's cache_dir (or CacheConfig.directory
for a direct system call). Both phase-censored views are processed by default;
--split inner warms only validation tuning. No labels, neighborhoods, or PCA
are precomputed. Rows are processed in chunks; existing vectors are reused on
restart. Use one writer per directory. Precomputation time remains part of the
method's runtime budget.

--batch-size controls GPU encoding batches (default 1024); reduce it if GPU
memory is insufficient. --chunk-rows controls CPU table-row chunks separately.

The default disk-vector cap matches the submission's 32 GiB cap; if increasing
--max-cache-gib, also increase text_cache_max_bytes in the submission config.
SQLite needs additional disk space beyond the vector payload. Encoding batches
differ from on-demand encoding, so FP16 results need not be bitwise identical.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from examples.kumo.relational._relarena.text import QwenDocuments
from examples.kumo.relational.rel_arena_model import (
    TEXT_TABLE_CHUNK_ROWS,
    table_stypes,
    v11_precision,
)
from relarena.dataset import RelBenchDatasetTask

import sdm


def main() -> None:
    """Fill the submission's raw-vector cache without loading the SDM model."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--split", choices=("inner", "outer", "both"), default="both"
    )
    parser.add_argument(
        "--chunk-rows",
        type=int,
        default=TEXT_TABLE_CHUNK_ROWS,
        help="CPU table-row chunk size (independent of GPU encoding batches)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="GPU encoding batch size; reduce if GPU memory is insufficient",
    )
    parser.add_argument("--max-cache-gib", type=int, default=32)
    args = parser.parse_args()
    if args.chunk_rows <= 0:
        parser.error("--chunk-rows must be positive")

    dataset = RelBenchDatasetTask(args.dataset, args.task, download=False)
    encoder = QwenDocuments(
        torch.device(args.device),
        cache_path=args.output_dir / "qwen-documents.sqlite",
        max_vector_bytes=args.max_cache_gib * 1024**3,
        batch_size=args.batch_size,
    )
    assert encoder.cache is not None
    phases = ("inner", "outer") if args.split == "both" else (args.split,)
    try:
        with v11_precision():
            for phase in phases:
                split = (
                    dataset.inner_split()
                    if phase == "inner"
                    else dataset.outer_split()
                )
                for name, table in sorted(split.db_state.table_dict.items()):
                    stypes = {
                        column: stype
                        for column, stype in table_stypes(
                            table, text=True
                        ).items()
                        if stype == sdm.Stype.text
                    }
                    if not stypes:
                        continue
                    for start in range(0, len(table.df), args.chunk_rows):
                        encoder.precompute(
                            sdm.TableTensor.from_pandas(
                                df=table.df.iloc[
                                    start : start + args.chunk_rows
                                ],
                                stypes=stypes,
                            )
                        )
                    print(
                        f"{phase}/{name}: {len(table.df):,} rows processed",
                        flush=True,
                    )
                del split
        print(f"Saved {encoder.cache.rows:,} embeddings in {args.output_dir}")
    finally:
        encoder.cache.close()


if __name__ == "__main__":
    main()
