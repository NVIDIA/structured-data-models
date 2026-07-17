"""
This script concatenates all text rows into a single text col. That column then gets embedded by the sentence transformer.

The main advantage to concatenation + embedding as one col is that the sentence transformer can capture cross-column relationships — e.g. if product: "credit card" and issue: "billing dispute" co-occur, the joint embedding can encode that interaction rather than treating them as independent signals.

The tradeoff is that individual column semantics can get diluted, especially if one column has much longer text than another and dominates the embedding. Embedding columns separately and concatenating would preserve per-column signal but lose those cross-column interactions (and multiply the embedding dimensionality by the number of columns).
"""
import argparse
import re
import sys
from collections import Counter
from pathlib import Path

# make the local sdm package importable
sys.path.insert(0, "/Users/jgagacheva/Documents/Dev/nvidia/structured-data-models")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap
from sdm.models import TabICLv2
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.random_projection import GaussianRandomProjection
from skrub.datasets import fetch_midwest_survey, fetch_toxicity, fetch_traffic_violations

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", choices=["traffic", "midwest", "toxicity"], default="traffic")
parser.add_argument(
    "--strable",
    help=(
        "Load a STRABLE benchmark classification table by name "
        "(e.g. financial-product-complaint). Overrides --dataset. "
        "See https://huggingface.co/datasets/inria-soda/STRABLE-benchmark"
    ),
)
parser.add_argument("--out-dir", default=".", help="Directory to write the heatmap PNG.")
parser.add_argument(
    "--cache-dir", default="emb_cache", help="Directory for cached per-encoder embeddings."
)
parser.add_argument(
    "--max-rows",
    type=int,
    default=40000,
    help="Subsample to at most this many rows (stratified) so TabICLv2 fits in GPU memory. 0 disables.",
)
parser.add_argument(
    "--supervised",
    action="store_true",
    help="Use supervised reducers (PLS, LDA, supervised UMAP) fit per CV fold on train rows + labels.",
)
parser.add_argument(
    "--encoders",
    help="Comma-separated encoders to run (default: all 4). Matches full id or basename.",
)
parser.add_argument(
    "--skip-full",
    action="store_true",
    help="Skip the full-dimension 'none' reducer (avoids the wide-embedding OOM); run only reduced dims.",
)
args = parser.parse_args()


def load_strable(name):
    """Load one STRABLE classification table as (X, y, text_col).

    All string columns are concatenated into a single synthetic '__text__'
    column ('colname: value | ...') that gets embedded; numeric columns are
    kept as extra tabular features. The target comes from the table's
    config.json.
    """
    import json
    import time

    import pandas as pd
    from huggingface_hub import hf_hub_download

    repo = "inria-soda/STRABLE-benchmark"

    def _dl(fn, tries=6):
        # retry with exponential backoff to ride out HF rate-limits (HTTP 429)
        for i in range(tries):
            try:
                return hf_hub_download(repo, fn, repo_type="dataset")
            except Exception as e:
                if i == tries - 1:
                    raise
                wait = 2 ** i
                print(f"  HF download {fn} failed ({type(e).__name__}); retry {i + 1}/{tries} in {wait}s")
                time.sleep(wait)

    cfg = json.load(open(_dl(f"{name}/config.json")))
    if "class" not in cfg.get("task", ""):
        raise SystemExit(
            f"STRABLE table {name!r} has task={cfg.get('task')!r}; this script only "
            "handles classification tables (TabICLv2 classifier)."
        )
    df = pd.read_parquet(_dl(f"{name}/data.parquet"))
    target = cfg["target_name"]
    df = df[df[target].notna()]
    y = df[target].reset_index(drop=True)
    X = df.drop(columns=[target]).reset_index(drop=True)
    # string/text columns, robust across pandas versions (object, string[pyarrow],
    # and the pandas>=3.0 'str' dtype); exclude anything numeric.
    str_cols = [
        c for c in X.columns
        if (pd.api.types.is_string_dtype(X[c]) or pd.api.types.is_object_dtype(X[c]))
        and not pd.api.types.is_numeric_dtype(X[c])
    ]
    if not str_cols:
        raise SystemExit(f"STRABLE table {name!r} has no string columns to embed")
    # each row's string cols are joined into a single string like: "col1: value1 | col2: value2 | ..." into a col called "__text__".
    text = X[str_cols].apply(
        lambda r: " | ".join(f"{c}: {r[c]}" for c in str_cols if pd.notna(r[c])), axis=1
    )
    X = X.drop(columns=str_cols)
    X["__text__"] = text
    print(f"STRABLE table: {name}  | target={target}  | string cols joined: {str_cols}")
    return X, y, "__text__"


if args.strable:
    X, y, text_col = load_strable(args.strable)
    safe = re.sub(r"[^A-Za-z0-9]+", "_", args.strable).strip("_")
    out_name = f"accuracy_heatmap_{safe}_tabiclv2.png"
    title = f"Downstream accuracy by encoder and reduction ({args.strable}, TabICLv2, 5-fold CV)"
elif args.dataset == "midwest":
    data = fetch_midwest_survey()
    text_col = "What_would_you_call_the_part_of_the_country_you_live_in_now"
    out_name = "accuracy_heatmap_midwest_tabiclv2.png"
    title = "Downstream accuracy by encoder and reduction (midwest_survey, TabICLv2, 5-fold CV)"
    X, y = data.X, data.y
elif args.dataset == "toxicity":
    data = fetch_toxicity()
    text_col = "text"
    out_name = "accuracy_heatmap_toxicity_tabiclv2.png"
    title = "Downstream accuracy by encoder and reduction (toxicity, TabICLv2, 5-fold CV)"
    X, y = data.X, data.y
else:
    data = fetch_traffic_violations()
    text_col = "description"
    out_name = "accuracy_heatmap_traffic_tabiclv2.png"
    title = "Downstream accuracy by encoder and reduction (traffic_violations, TabICLv2, 5-fold CV)"
    X, y = data.X, data.y

desc = X[text_col].dropna()
lengths = desc.str.split().str.len()
top10 = Counter(re.findall(r"[a-z]+", " ".join(desc.str.lower()))).most_common(10)
print(f"description column stats:")
print(f"  rows:          {len(desc)}")
print(f"  unique values: {desc.nunique()}")
print(f"  avg words:     {lengths.mean():.1f}")
print(f"  min/max words: {lengths.min()} / {lengths.max()}")
print(f"  top 10 words:  {', '.join(w for w, _ in top10)}")

# sentence transformer models to benchmark
ENCODERS = [
    "all-MiniLM-L6-v2",
    "intfloat/e5-small-v2",
    "BAAI/bge-base-en-v1.5",
    "nvidia/llama-nemotron-embed-1b-v2",
]
if args.encoders:
    wanted = {e.strip() for e in args.encoders.split(",")}
    ENCODERS = [e for e in ENCODERS if e in wanted or e.split("/")[-1] in wanted]
    if not ENCODERS:
        raise SystemExit(f"--encoders matched no known encoder: {args.encoders!r}")

# dimensionality reduction strategies applied to the embeddings before feeding into the model
# "none" keeps the full embedding dimension; "truncate" only makes sense for MRL models
REDUCERS = {
    "none": lambda k: None,
    "pca": lambda k: PCA(n_components=k),
    "randproj": lambda k: GaussianRandomProjection(n_components=k, random_state=0),
    "truncate": lambda k: "truncate",
}


# --- supervised reducers: fit per CV fold on train-row embeddings + labels (leakage-free) ---
class _PLSReducer:
    """Partial Least Squares against one-hot labels; keeps k covariance-max directions."""

    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        from sklearn.cross_decomposition import PLSRegression

        classes = np.unique(y)
        Y = (y[:, None] == classes[None, :]).astype(float)  # one-hot target
        self._m = PLSRegression(n_components=min(self.k, X.shape[1])).fit(X, Y)
        return self

    def transform(self, X):
        return self._m.transform(X)


class _UMAPReducer:
    """Supervised UMAP: fit warps the manifold toward class structure via y."""

    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        import umap

        self._m = umap.UMAP(n_components=self.k, random_state=0).fit(X, y)
        return self

    def transform(self, X):
        return self._m.transform(X)


def _make_lda(k, n_classes):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    # LDA yields at most n_classes-1 components
    return LinearDiscriminantAnalysis(n_components=max(1, min(k, n_classes - 1)))


SUPERVISED_REDUCERS = {
    "pls": lambda k, nc: _PLSReducer(k),
    "lda": lambda k, nc: _make_lda(k, nc),
}
try:  # supervised UMAP only if umap-learn is installed
    import umap  # noqa: F401

    SUPERVISED_REDUCERS["umap_sup"] = lambda k, nc: _UMAPReducer(k)
except Exception:
    pass

# target embedding dimensions to try for each reducer (independent of number of classes)
DIMS = [16, 30]
N_SPLITS = 5

# identifier used for cache filenames and output; unique per dataset / strable table
dataset_id = re.sub(r"[^A-Za-z0-9]+", "_", args.strable or args.dataset).strip("_")
cache_dir = Path(args.cache_dir)
cache_dir.mkdir(parents=True, exist_ok=True)
out_dir = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

# drop rows where the text column is missing so np.unique can sort cleanly
mask = X[text_col].notna()
X, y = X[mask], y[mask]

# subsample very large datasets so TabICLv2's in-context context fits in GPU memory
# (in-context models materialize the whole training context). stratified, fixed seed.
if args.max_rows and len(X) > args.max_rows:
    from sklearn.model_selection import train_test_split

    try:
        splits = train_test_split(X, y, train_size=args.max_rows, stratify=y, random_state=0)
    except ValueError:
        splits = train_test_split(X, y, train_size=args.max_rows, random_state=0)
    X, y = splits[0], splits[2]
    print(f"subsampled to {len(X)} rows (max_rows={args.max_rows}) to fit TabICLv2 memory")

# deduplicate text values so we only encode each unique string once,
# then use `codes` to expand embeddings back to one row per sample
# whole cell embedding
uniques, codes = np.unique(X[text_col], return_inverse=True)

# keep only numeric columns from the non-text features
X_rest = X.drop(columns=[text_col]).select_dtypes("number")

# TabICLv2 requires integer class labels, so encode the string target
le = LabelEncoder()
y_encoded = le.fit_transform(y)
n_classes = len(le.classes_)
# majority-class baseline: accuracy from always predicting the most frequent label.
# any model worth keeping must beat this; report lift = score - baseline below.
majority_baseline = np.bincount(y_encoded).max() / len(y_encoded)
title = f"{title}  (majority baseline {majority_baseline:.3f})"
print(f"majority-class baseline accuracy: {majority_baseline:.4f}  (n_classes={n_classes})")
# TabICLv2 supports at most 10 classes (the classifier errors beyond that)
if n_classes > 10:
    raise SystemExit(
        f"{n_classes} classes found; TabICLv2 supports at most 10. "
        "Choose a table/target with <=10 classes (see list_strable.py)."
    )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# load pretrained TabICLv2 weights from HuggingFace (jingang/TabICL)
tabiclv2 = TabICLv2(device=device)

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=0)

rows = {}  # encoder -> {column_label: score}
records = []  # flat per-config metrics for CSV export

for enc_name in ENCODERS:
    # cache embeddings to disk so re-runs skip the expensive encode step
    cache = cache_dir / f"emb_{dataset_id}_{enc_name.replace('/', '_')}.npy"
    if cache.exists():
        emb = np.load(cache)
    else:
        model = SentenceTransformer(
            enc_name,
            trust_remote_code=True,
            model_kwargs={"torch_dtype": "bfloat16"},
        )
        # encode only unique strings, not all rows
        # The numeric columns are kept separately and concatenated with the embeddings just before feeding into TabICLv2.
        emb = model.encode(
            list(uniques),
            prompt_name="document" if "nemotron" in enc_name else None,
        )
        np.save(cache, emb)

    emb_rows = emb[codes]  # (n_rows, dim) row-space embeddings
    X_rest_np = X_rest.to_numpy()
    reducers = SUPERVISED_REDUCERS if args.supervised else dict(REDUCERS)
    if args.skip_full:
        reducers.pop("none", None)  # drop the full-dimension config (wide-embedding OOM)

    for red_name, make in reducers.items():
        # "none" uses the full embedding dim; others iterate over DIMS
        for k in DIMS if red_name != "none" else [emb.shape[1]]:
            if not args.supervised:
                # unsupervised: fit once on all unique embeddings, then expand to rows
                reducer = make(k)
                if reducer == "truncate":
                    reduced_u = emb[:, :k]
                elif reducer is None:
                    reduced_u = emb
                else:
                    reduced_u = reducer.fit_transform(emb)
                reduced_rows = reduced_u[codes]

            fold_scores = []
            fold_f1s = []
            for train_idx, test_idx in skf.split(emb_rows, y_encoded):
                if args.supervised:
                    # fit reducer per fold on TRAIN rows + labels only (no leakage)
                    red = make(k, n_classes)
                    red.fit(emb_rows[train_idx], y_encoded[train_idx])
                    red_tr = np.asarray(red.transform(emb_rows[train_idx]))
                    red_te = np.asarray(red.transform(emb_rows[test_idx]))
                else:
                    red_tr = reduced_rows[train_idx]
                    red_te = reduced_rows[test_idx]

                # concatenate reduced embedding with the numeric features, per split
                X_tr = np.hstack([X_rest_np[train_idx], red_tr])
                X_te = np.hstack([X_rest_np[test_idx], red_te])
                x_train = torch.tensor(X_tr, dtype=torch.float32).to(device)
                x_test = torch.tensor(X_te, dtype=torch.float32).to(device)
                y_train = torch.tensor(y_encoded[train_idx], dtype=torch.int64).to(device)
                y_test = torch.tensor(y_encoded[test_idx], dtype=torch.int64).to(device)

                # TabICLv2 is an in-context model: fit caches the training
                # context, predict runs inference on test rows, clear frees it
                with torch.amp.autocast(device.type, torch.bfloat16):
                    tabiclv2.fit(x=x_train, y=y_train)
                    logits = tabiclv2.predict(x=x_test)
                    tabiclv2.clear()

                preds = logits.argmax(dim=-1)
                acc = (preds == y_test).float().mean().item()
                fold_f1 = f1_score(
                    y_encoded[test_idx], preds.cpu().numpy(), average="macro"
                )
                fold_scores.append(acc)
                fold_f1s.append(fold_f1)

            score = np.mean(fold_scores)
            macro_f1 = np.mean(fold_f1s)
            lift = score - majority_baseline
            print(
                f"{enc_name:35s} {red_name:9s} k={k:5d}  "
                f"acc={score:.4f}  macroF1={macro_f1:.4f}  lift={lift:+.4f}"
            )
            col = "full" if red_name == "none" else f"{red_name}\nk={k}"
            rows.setdefault(enc_name, {})[col] = score
            records.append({
                "dataset": dataset_id,
                "encoder": enc_name,
                "reducer": red_name,
                "k": k,
                "acc": round(float(score), 4),
                "macro_f1": round(float(macro_f1), 4),
                "lift": round(float(lift), 4),
                "baseline": round(float(majority_baseline), 4),
                "n_classes": n_classes,
            })

# write per-config metrics to CSV alongside the heatmap
import csv as _csv

csv_path = out_dir / f"results_{dataset_id}.csv"
with open(csv_path, "w", newline="") as _f:
    _w = _csv.DictWriter(
        _f,
        fieldnames=["dataset", "encoder", "reducer", "k", "acc", "macro_f1", "lift", "baseline", "n_classes"],
    )
    _w.writeheader()
    _w.writerows(records)
print(f"wrote {csv_path}")

# ── plotting ────────────────────────────────────────────────────────────────

RAMP = [
    "#cde2fb",
    "#b7d3f6",
    "#9ec5f4",
    "#86b6ef",
    "#6da7ec",
    "#5598e7",
    "#3987e5",
    "#2a78d6",
    "#256abf",
    "#1c5cab",
    "#184f95",
    "#104281",
    "#0d366b",
]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#898781"

encoders = list(rows)
columns = list(rows[encoders[0]])
grid = np.array([[rows[e][c] for c in columns] for e in encoders])

short = {e: e.split("/")[-1] for e in encoders}
cmap = LinearSegmentedColormap.from_list("seq_blue", RAMP)

fig, ax = plt.subplots(
    figsize=(1.35 * len(columns) + 2.5, 0.9 * len(encoders) + 1.6),
)
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)
im = ax.imshow(grid, cmap=cmap, aspect="auto")

ax.set_xticks(np.arange(-0.5, len(columns)), minor=True)
ax.set_yticks(np.arange(-0.5, len(encoders)), minor=True)
ax.grid(which="minor", color=SURFACE, linewidth=2)
ax.tick_params(which="both", length=0)

best = grid.max()
norm_mid = grid.min() + 0.55 * (grid.max() - grid.min())
for i in range(len(encoders)):
    for j in range(len(columns)):
        v = grid[i, j]
        ax.text(
            j,
            i,
            f"{v:.3f}".lstrip("0"),
            ha="center",
            va="center",
            fontsize=10,
            color="#ffffff" if v > norm_mid else INK,
            fontweight="bold" if v == best else "normal",
        )

ax.set_xticks(range(len(columns)), labels=columns, color=INK)
ax.set_yticks(range(len(encoders)), labels=[short[e] for e in encoders], color=INK)
for spine in ax.spines.values():
    spine.set_visible(False)
ax.set_title(title, loc="left", color=INK, pad=14)
cbar = fig.colorbar(im, ax=ax, shrink=0.8)
cbar.outline.set_visible(False)
cbar.ax.tick_params(color=MUTED, labelcolor=MUTED)

fig.tight_layout()
out = out_dir / out_name
fig.savefig(out, dpi=150, facecolor=SURFACE)
print(f"wrote {out}")
