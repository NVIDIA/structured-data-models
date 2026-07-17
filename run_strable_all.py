#!/usr/bin/env python3
"""Run text_embed_tabiclv2.py --strable on every STRABLE classification table.

Saves each heatmap into --out-dir. Skips tables whose heatmap already exists,
and continues past any table that fails (logs go next to the heatmaps).

Usage (on the VM):
    ~/tabicl-venv/bin/python ~/run_strable_all.py
    ~/tabicl-venv/bin/python ~/run_strable_all.py --max-rows 30000
    ~/tabicl-venv/bin/python ~/run_strable_all.py --only lending-club-loan osha-accidents
"""
import argparse
import json
import os
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

REPO = "inria-soda/STRABLE-benchmark"
TREE = f"https://huggingface.co/api/datasets/{REPO}/tree/main?recursive=false"
BASE = f"https://huggingface.co/datasets/{REPO}/resolve/main"
CACHE = Path.home() / ".strable_classification_tables.json"


def _get_json(url, tries=6):
    """Fetch JSON with exponential backoff so HF rate-limits (429) self-recover."""
    import time
    import urllib.error

    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 429 and i < tries - 1:
                wait = 5 * (i + 1)
                print(f"  HF API 429; retry {i + 1}/{tries} in {wait}s")
                time.sleep(wait)
                continue
            raise
        except Exception:
            if i < tries - 1:
                time.sleep(2 ** i)
                continue
            raise


def classification_tables():
    # cache the discovered list so we hit the API once, not 100+ times per run
    if CACHE.exists():
        try:
            return json.loads(CACHE.read_text())
        except Exception:
            pass
    dirs = [x["path"] for x in _get_json(TREE) if x.get("type") == "directory"]
    out = []
    for d in dirs:
        try:
            if "class" in _get_json(f"{BASE}/{d}/config.json").get("task", ""):
                out.append(d)
        except Exception:
            pass
    out = sorted(out)
    try:
        CACHE.write_text(json.dumps(out))
    except Exception:
        pass
    return out


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="results -- concatenated cols")
    ap.add_argument("--max-rows", type=int, default=20000,
                    help="Row cap per table (20k is safe for the 2048-dim encoder on a 48GB GPU).")
    ap.add_argument("--script", default=str(here / "text_embed_tabiclv2.py"))
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--only", nargs="*", help="Run only these table names.")
    ap.add_argument("--supervised", action="store_true", help="Use supervised reducers (pass through).")
    ap.add_argument("--encoders", help="Comma-separated encoders to run (pass through).")
    ap.add_argument("--skip-full", action="store_true", help="Skip full-dim 'none' reducer (pass through).")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tables = args.only or classification_tables()
    print(f"{len(tables)} tables -> {out}/  (max_rows={args.max_rows})\n")

    summary = []
    for i, t in enumerate(tables, 1):
        safe = re.sub(r"[^A-Za-z0-9]+", "_", t).strip("_")
        png = out / f"accuracy_heatmap_{safe}_tabiclv2.png"
        if png.exists():
            print(f"[{i}/{len(tables)}] skip {t} (already done)")
            summary.append((t, "skip"))
            continue
        print(f"[{i}/{len(tables)}] run  {t}")
        log = out / f"log_{safe}.txt"
        emb = out / "emb_cache"
        emb.mkdir(parents=True, exist_ok=True)
        env = dict(
            os.environ,
            PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
            HF_HUB_DISABLE_XET="1",  # avoid the rate-limited xet download path
        )
        cap = args.max_rows
        status = None
        while True:
            # row count changes the unique-text set, so clear this table's stale cache first
            for p in emb.glob(f"emb_{safe}_*.npy"):
                p.unlink()
            cmd = [
                args.python, "-u", args.script, "--strable", t,
                "--out-dir", str(out), "--cache-dir", str(emb), "--max-rows", str(cap),
            ]
            if args.supervised:
                cmd.append("--supervised")
            if args.encoders:
                cmd += ["--encoders", args.encoders]
            if args.skip_full:
                cmd.append("--skip-full")
            with open(log, "w") as f:
                rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env).returncode
            if rc == 0 and png.exists():
                status = f"ok (max_rows={cap})"
                break
            # on CUDA OOM, halve the rows and retry until it fits (floor 4000)
            if "OutOfMemoryError" in log.read_text() and cap > 4000:
                cap //= 2
                print(f"        OOM -> retrying {t} at max_rows={cap}")
                continue
            status = f"FAIL(rc={rc})"
            break
        print(f"        -> {status}   (log: {log})")
        summary.append((t, status))

    # merge all per-table results_*.csv into one combined file
    combined = out / "results_all.csv"
    csvs = sorted(p for p in out.glob("results_*.csv") if p.name != "results_all.csv")
    if csvs:
        with open(combined, "w", newline="") as fout:
            for i, p in enumerate(csvs):
                lines = p.read_text().splitlines()
                fout.write("\n".join(lines if i == 0 else lines[1:]) + "\n")
        print(f"\ncombined metrics -> {combined}")

    print("\n=== summary ===")
    for t, s in summary:
        print(f"  {s:12s} {t}")
    ok = sum(1 for _, s in summary if s == "ok")
    print(f"\n{ok}/{len(summary)} tables produced heatmaps in {out}/")


if __name__ == "__main__":
    main()
