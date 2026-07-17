#!/usr/bin/env python3
"""Survey the STRABLE benchmark classification tables for TabICL suitability.

For each classification table it reports: task (binary/multi), #classes,
majority baseline, rows, #string cols, #numeric cols, and the average word
count of the richest string column. Flags tables that fit TabICLv2
(2-10 classes). Prints a table sorted by text richness so you can pick a
good one for text_embed_tabiclv2.py --strable <name>.

Usage:
    python list_strable.py                # all 32 classification tables
    python list_strable.py --all          # include regression tables too
    python list_strable.py --max-rows 5   # cap rows read per table (faster)
"""
import argparse
import json
import urllib.request

import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

REPO = "inria-soda/STRABLE-benchmark"
# HuggingFace API endpoint to list top-level directories (one per table)
TREE = f"https://huggingface.co/api/datasets/{REPO}/tree/main?recursive=false"


def list_tables():
    # Fetch the repo tree and return sorted directory names (each is one dataset table)
    with urllib.request.urlopen(TREE, timeout=30) as r:
        tree = json.load(r)
    return sorted(x["path"] for x in tree if x.get("type") == "directory")


def inspect(name):
    # Download config to learn the task type (classification/regression) and target column
    cfg = json.load(open(hf_hub_download(REPO, f"{name}/config.json", repo_type="dataset")))
    task = cfg.get("task", "?")
    target = cfg.get("target_name")

    # Read only the parquet schema and metadata to avoid loading all data upfront
    path = hf_hub_download(REPO, f"{name}/data.parquet", repo_type="dataset")
    schema = pq.read_schema(path)
    n_rows = pq.ParquetFile(path).metadata.num_rows

    # Split columns into string vs numeric; exclude target from numeric count
    str_cols = [f.name for f in schema if str(f.type) in ("string", "large_string")]
    num_cols = [f.name for f in schema if f.name != target and f.name not in str_cols]

    # Load only string cols + target to keep memory usage low
    cols = list(dict.fromkeys([*str_cols, target]))
    df = pd.read_parquet(path, columns=cols)
    df = df[df[target].notna()]  # drop rows with missing labels

    # Compute class count and majority-class baseline accuracy
    n_classes = df[target].nunique()
    majority = df[target].value_counts(normalize=True).iloc[0] if len(df) else float("nan")

    # Find the string column with the highest average word count (richest text signal)
    best_words, best_col = 0.0, None
    for c in str_cols:
        if c == target:
            continue
        w = df[c].dropna().astype(str).str.split().str.len().mean()
        if pd.notna(w) and w > best_words:
            best_words, best_col = w, c

    return {
        "name": name, "task": task, "n_classes": int(n_classes),
        "majority": round(float(majority), 3), "rows": int(n_rows),
        "n_str": len(str_cols), "n_num": len(num_cols),
        "text_words": round(best_words, 1), "text_col": best_col,
        "tabicl_ok": 2 <= n_classes <= 10,  # TabICLv2 supports 2-10 classes
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="include regression tables")
    args = ap.parse_args()

    # Inspect each table, skipping ones that error or aren't classification tasks
    rows = []
    for name in list_tables():
        try:
            info = inspect(name)
        except Exception as e:
            print(f"  skip {name}: {type(e).__name__}: {str(e)[:60]}")
            continue
        if not args.all and "class" not in info["task"]:
            continue
        rows.append(info)

    # Sort: TabICL-compatible tables first, then by text richness descending
    rows.sort(key=lambda r: (r["tabicl_ok"], r["text_words"]), reverse=True)

    # Print aligned summary table
    hdr = f'{"table":40s} {"task":16s} {"cls":>3s} {"base":>5s} {"rows":>8s} {"str":>3s} {"num":>3s} {"words":>6s}  fit  text_col'
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in rows:
        fit = "OK " if r["tabicl_ok"] else "no "
        print(f'{r["name"]:40s} {r["task"]:16s} {r["n_classes"]:3d} {r["majority"]:5.2f} '
              f'{r["rows"]:8d} {r["n_str"]:3d} {r["n_num"]:3d} {r["text_words"]:6.1f}  {fit} {r["text_col"]}')

    # Print top candidates: fit for TabICLv2 and richest text column
    ok = [r for r in rows if r["tabicl_ok"]]
    print(f"\n{len(ok)}/{len(rows)} classification tables fit TabICLv2 (2-10 classes).")
    print("Good text candidates (fit + longest text):")
    for r in sorted(ok, key=lambda r: r["text_words"], reverse=True)[:8]:
        print(f'  --strable {r["name"]:40s} ({r["n_classes"]} cls, {r["text_words"]:.0f} words, {r["rows"]} rows)')


if __name__ == "__main__":
    main()
