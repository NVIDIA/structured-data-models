#!/usr/bin/env python3
"""Side-by-side comparison of reduction methods per table, for Nemotron.

Merges the unsupervised results (results -- concatenated cols/results_all.csv)
and the supervised results (results -- supervised/results_all.csv), keeps only
the Nemotron rows, takes each method's best-k, and renders a single master
heatmap: rows = tables, columns = reduction methods, cell = LIFT over the
majority-class baseline.

Color is a diverging blue<->red scale centered at 0 (blue = above baseline /
the method helps, gray = at baseline, red = below baseline / it hurts) so the
tables where supervised reduction backfires are immediately legible.

Also writes a tidy merged CSV with the numbers behind the picture.

Usage:
    python compare_reducers.py \
        --unsup "results -- concatenated cols/results_all.csv" \
        --sup   "results -- supervised/results_all.csv" \
        --out-dir results
"""
import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

# palette (validated diverging pair + chrome) from the dataviz reference
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
BLUE = "#184f95"   # positive pole: method beats baseline
RED = "#c62f2f"    # negative pole: method below baseline
MIDGRAY = "#f0efec"  # neutral midpoint at lift = 0

NEMO = "nvidia/llama-nemotron-embed-1b-v2"
# unsupervised first, then a gap, then supervised
METHOD_ORDER = ["none", "pca", "randproj", "truncate", "pls", "lda", "umap_sup"]
SUPERVISED = {"pls", "lda", "umap_sup"}
PRETTY = {"none": "full", "umap_sup": "umap"}


def load(path):
    rows = []
    for r in csv.DictReader(open(path)):
        if r["encoder"] != NEMO:
            continue
        r["acc"] = float(r["acc"])
        r["lift"] = float(r["lift"])
        r["baseline"] = float(r["baseline"])
        r["k"] = int(r["k"])
        r["n_classes"] = int(r["n_classes"])
        rows.append(r)
    return rows


def best_by_method(rows):
    """table -> method -> (best-k row by accuracy)."""
    out = {}
    for r in rows:
        d = out.setdefault(r["dataset"], {})
        cur = d.get(r["reducer"])
        if cur is None or r["acc"] > cur["acc"]:
            d[r["reducer"]] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unsup", default="results -- concatenated cols/results_all.csv")
    ap.add_argument("--sup", default="results -- supervised/results_all.csv")
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    rows = []
    for p in (args.unsup, args.sup):
        if Path(p).exists():
            rows += load(p)
    if not rows:
        raise SystemExit("no Nemotron rows found in either CSV")

    by = best_by_method(rows)
    tables = sorted(by, key=lambda t: next(iter(by[t].values()))["baseline"])  # balanced -> imbalanced
    methods = [m for m in METHOD_ORDER if any(m in by[t] for t in tables)]

    # lift matrix (NaN where a method wasn't run for a table)
    M = np.full((len(tables), len(methods)), np.nan)
    for i, t in enumerate(tables):
        for j, m in enumerate(methods):
            if m in by[t]:
                M[i, j] = by[t][m]["lift"]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- tidy merged CSV ---
    csv_path = out_dir / "reducer_comparison_nemotron.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["table", "method", "best_k", "acc", "macro_f1", "lift", "baseline", "supervised"])
        for t in tables:
            for m in methods:
                if m in by[t]:
                    r = by[t][m]
                    w.writerow([t, m, r["k"], f"{r['acc']:.4f}", r["macro_f1"],
                                f"{r['lift']:.4f}", f"{r['baseline']:.4f}",
                                "y" if m in SUPERVISED else "n"])
    print(f"wrote {csv_path}")

    # --- diverging heatmap ---
    cmap = LinearSegmentedColormap.from_list("div_bluered", [RED, MIDGRAY, BLUE])
    cmap.set_bad(SURFACE)  # NaN cells fade into the surface
    span = np.nanmax(np.abs(M)) or 0.1
    norm = TwoSlopeNorm(vmin=-span, vcenter=0.0, vmax=span)

    fig, ax = plt.subplots(figsize=(1.15 * len(methods) + 3.5, 0.5 * len(tables) + 2.2))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    im = ax.imshow(M, cmap=cmap, norm=norm, aspect="auto")

    # 2px surface gap between cells
    ax.set_xticks(np.arange(-0.5, len(methods)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(tables)), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="both", length=0)

    # annotate lift; white ink on saturated cells, dark ink near the neutral middle
    for i in range(len(tables)):
        for j in range(len(methods)):
            v = M[i, j]
            if np.isnan(v):
                ax.text(j, i, "–", ha="center", va="center", color=MUTED, fontsize=9)
                continue
            frac = abs(norm(v) - 0.5) * 2  # 0 at midpoint, 1 at a pole
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=9,
                    color="#ffffff" if frac > 0.55 else INK)

    # column labels: unsupervised vs supervised marked
    col_labels = [PRETTY.get(m, m) for m in methods]
    ax.set_xticks(range(len(methods)), labels=col_labels, color=INK)
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    for tick, m in zip(ax.get_xticklabels(), methods):
        if m in SUPERVISED:
            tick.set_color(BLUE)
            tick.set_fontweight("bold")

    # row labels: table + its baseline; imbalanced tables (majority class >=1.5x its
    # fair share = base * n_classes >= 1.5) marked in bold red
    def _imbalanced(t):
        r = next(iter(by[t].values()))
        return r["baseline"] * r["n_classes"] >= 1.5

    ylabels = [f"{t}  (base {next(iter(by[t].values()))['baseline']:.2f})" for t in tables]
    ax.set_yticks(range(len(tables)), labels=ylabels, fontsize=9)
    for lab, t in zip(ax.get_yticklabels(), tables):
        if _imbalanced(t):
            lab.set_color(RED)
            lab.set_fontweight("bold")
        else:
            lab.set_color(INK)

    # divider between unsupervised and supervised column groups
    n_unsup = sum(1 for m in methods if m not in SUPERVISED)
    if 0 < n_unsup < len(methods):
        ax.axvline(n_unsup - 0.5, color=MUTED, linewidth=1.5)

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(
        "Nemotron: lift over baseline by reduction method\n"
        "cells: blue = beats baseline, red = below · columns: bold = supervised\n"
        "rows sorted by class balance · bold-red table name = imbalanced (majority class ≥ 1.5× fair share)",
        loc="left", color=INK, pad=34, fontsize=11,
    )
    cbar = fig.colorbar(im, ax=ax, shrink=0.7)
    cbar.set_label("lift over majority baseline", color=MUTED)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(color=MUTED, labelcolor=MUTED)

    fig.tight_layout()
    png = out_dir / "reducer_comparison_nemotron.png"
    fig.savefig(png, dpi=150, facecolor=SURFACE)
    print(f"wrote {png}  ({len(tables)} tables x {len(methods)} methods)")


if __name__ == "__main__":
    main()
