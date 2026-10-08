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

## Completed replacement-host staging

On the replacement host1, all 32 real-data manifest members passed independent host-side SHA256 verification. The transfer contained 368,889,517 bytes including manifests and auxiliary metadata, compressed to approximately 181.9 MB, and took approximately 138 seconds at the reported rsync rate. The fixed C16k/C64k workload transfer contained 198,050,765 bytes, compressed to 54.5 MB, and took approximately 48 seconds. Graph, workload-metadata and validation-label hashes on the host matched the authorities above. These are measured transfer outcomes, superseding the earlier bandwidth scenarios for this host; they are not inference timings.

The C16k directory also contains `workload-metadata-correction.json`: macOS `resource.ru_maxrss` was originally labeled KiB even though macOS returns bytes. The original `max_rss_kib=4485513216` corresponds to corrected `4380384` KiB. The sidecar preserves the original metadata and all graph bytes; this is a unit correction, not changed sampling.

The retained F1 C1024/B125 workload was also transferred and its four files verified on host1, ready for copying to a replacement peer. Its `graphs.pt` SHA256 is `23c439d0ff2771af5fcaed8735607d1fe762ee2236bb5451eceda7cfdf1adcb2`, `workload.json` SHA256 is `75a5ca5d492668ad5ceb7e254c219008380ff73a41f6e1d298845461007af307`, and validation-label SHA256 is `d79603b42e771aefb580b07a1e41c9256f0fd6eacb13d34e7b22b3a17b8451cd`. Total size is 5,073,502 bytes. Conversely, the historical H&M C1024/Q2000/B250 graph binary was not retained locally: only its workload metadata was found. Any replacement small-H&M preparation must be marked fresh and shared across its new matched controls; metadata resemblance alone does not establish historical graph-byte identity.

The initial `source-c0fb64fcc` remains unchanged. A **second**, independently archived source `b721583a9f2ffe4eaf0931b61cdaa9340c8bbe8f` is staged at `source-b721583a9` for later DP repeat-retention and corrected GNN-buffer-lifetime comparisons. Its archive SHA256 is `281e6e4e632fadc7f16113121698261682be04d56d0b489e26d3d3f34b5b7265`; the package-manifest SHA256 is `07212b1dee668ebfec6bbbb7f0bb6b06c366a1210b277380242d74fe4a66a42f`. Both source archives passed host-side checksum verification before fresh extraction. Do not silently compare different revisions as though source were held fixed.

Checkpoint installation remains the operator's separate responsibility: exact pinned Hugging Face revisions are downloaded directly to the host. Dataset/source staging success does not imply host model verification or a completed GPU experiment.

The replacement host2 (Frankfurt, four L4 GPUs) subsequently received both frozen source archives, Covertype/California/F1 data and the exact retained F1/C16k/C64k workloads. Both archives passed host-side checksum verification before fresh extraction. All 25 staged raw-data members and all three dataset-manifest digests matched local authorities; all 13 sampled-workload files, including the C16k RSS correction sidecar, matched independent local SHA256 digests. The payload totaled 363,423,846 bytes and compressed to approximately 82.5 MB; the three concurrent rsync streams completed in approximately 31 seconds after SSH became ready. No model files were included in this transfer. Raw H&M parquet was intentionally omitted because fixed-graph replay does not need it. Input transfer and verification ownership was explicitly released before timing; the operator separately gates runtime/checkpoint readiness and runners gate GPU leases.
