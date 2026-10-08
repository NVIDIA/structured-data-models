# Replacement-host replay readiness

Audit date: 2026-10-08. This is an input-integrity and transfer-readiness note, not a GPU result. All paths below are relative to `/Users/ardrianw/repositories/.kumo-multigpu-20261008` locally and should retain their portable suffix on the host.

## Rechecked authorities

All 15 files in `models/manifest.json` and all 32 files across the four `data/*/manifest.json` files passed fresh SHA256 verification. Model manifest members total 2,681,587,284 bytes; real-data members total 368,870,119 bytes. Exclude `data/source_cache/` (approximately 14 MiB), which is not required for replay. Model revisions and licenses remain those in [workloads.md](workloads.md); no checkpoint download, dataset resampling or label regeneration is needed.

The frozen runtime source is **`c0fb64fcc5c062c98b5fb47d1683f63757931fa9`**, not the initially requested `bd7e1ba40`. The coordinator approved this newer freeze after runner fixes. `ops/source-c0fb64fcc-bundle/` contains the exact `git archive` and a sorted `SHA256SUMS`; it was copied into a fresh read-only package and verified. The archive is 11,612,160 bytes, with SHA256 `173520b99b5a6a4b087b081cb8632b122f171a8191a69e238a3cd8faf8f0a853`. The package manifest SHA256 is `a14bfa5ba3220dcc35876d40b0608504bcfd3b43a5be2d10bdf1bd6c844c883f`. No mutable working tree or credentials are included.

## Exact native H&M replay inputs

Both runner workloads contain 4,096 identical ordered VAL task rows and labels, eight query batches of 512, temporal `last` sampling, exact `[16,16]` fanouts and sampler seed **1729**. Context row IDs equal prefixes of the shared TRAIN permutation prepared with seed 20261008. Sampler seed and TRAIN permutation seed are different authorities.

| Workload directory under `results/relational/` | Context nodes | Directed edges | `graphs.pt` SHA256 |
|---|---:|---:|---|
| `workload-hm-c16384-b512` | 319,383 | 641,272 | `79df998020561488bb4cb18e6ec8fc81090e69209ede50eebf9f8d4b8add4f3d` |
| `workload-hm-c65536-b512` | 1,279,607 | 2,571,320 | `1176d9e75c8f019920cce1f8887d1360f518de254e7850af8a45f9f4dc2183a6` |

An independent CPU deserialization audit recomputed every task/table digest, row count, relationship and task-link representation for context and all query batches, matching each workload manifest. It also checked context-ID prefix identity and rebuilt `TaskGraph` to measure node/edge counts. C64k tables contain 571,241 article, 65,536 customer and 642,830 transaction rows, all with valid task ownership. Per-table column-attention key selection can therefore reach its 20,000-key cap; preserve member RNG and selected-key identities across exact arms.

The C64k `workload.json` SHA256 is `1430f209578aff6614ea4d8e24d2791b31e8d26b97ad2f6f99a3594887dd1f55`; C16k is `4379416dd97f13fbd46db1d2c692a80529e589a60a00699394dc8baedaea6235`. Both `validation-labels.pt` files have SHA256 `b80778a026099c78e2fad68fdd50cc23cf161f4c9b5b663726f91bcf4ac76c40`.

### Two important non-equivalences

1. Earlier `cpu-profiles/rel-hm-c65536-q256` diagnostics used sampler seed 20261008 and contain **1,279,622** nodes, including 571,256 articles. They are not the runner's C64k graph. The GNN shape estimate still rounds to 6.10 GiB, but do not substitute that artifact for fresh blocked/unblocked comparisons.
2. C16k and C64k runner workloads have identical query task identities but **different sampled query neighborhoods**. Every query batch has different transaction/article hashes, and article counts differ slightly, because context sampling advances the RNG before query sampling. They support matched-method comparisons within a fixed workload, but do not isolate a pure context-size quality effect across workloads. Use a shared query graph artifact in a future pure context-size study; do not rewrite existing evidence.

All destination-block, placement and native controls at C64k must read the **same existing** `workload-hm-c65536-b512/graphs.pt`; never invoke `prepare` independently per arm. Block size must change execution only, not sampling, context membership, dtype, member seed or ensemble plan.

## Minimal transfer and verification

The full portable model cache plus four real datasets is about 2.84 GiB before source and sampled graphs. A relational-only replay needs approximately 432 MiB: relational checkpoint cache (232 MiB), C64k workload (131 MiB), C16k workload (58 MiB), and source archive (11 MiB). Raw relational parquet is unnecessary when replaying these already sampled graphs, but retain it when new preparation is explicitly in scope. The full model manifest includes tabular members: if transferring only relational weights, verify that subset explicitly rather than claiming the complete manifest was staged.

Use `rsync -a` with explicit source directories, no `--delete`, and preserve the model cache's `refs/` and snapshot layout. Stage the source bundle independently, then on the host:

```sh
cd /home/ubuntu/kumo-multigpu/source-c0fb64fcc-bundle
sha256sum -c SHA256SUMS
mkdir /home/ubuntu/kumo-multigpu/source-c0fb64fcc
tar -xf source-c0fb64fcc.tar -C /home/ubuntu/kumo-multigpu/source-c0fb64fcc
```

The extraction destination must be fresh. If it exists, verify and reuse the approved existing source or choose a new destination; do not overwrite a possibly modified tree. Independently verify model and graph hashes on the host before inference. Set `HF_HUB_CACHE=/home/ubuntu/kumo-multigpu/models/hub`, `HF_HUB_OFFLINE=1` and `PYTHONPATH` to that exact extracted source before importing SDM.

Transfer-only feasibility estimate: at an observed 10–50 MiB/s, the relational-only payload would take roughly 9–43 seconds; full model/data transfer roughly 1–5 minutes. These are bandwidth scenarios, not measured transfer times, and exclude provisioning, Python installation, hash verification and model warmup. Actual operator receipts determine what reached a replacement host. This local readiness audit does not assert a successful remote transfer.
