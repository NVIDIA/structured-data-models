# TabFM L4 ensemble-parallel inference diagnostic

Date: 2026-08-27

Status: **exploratory, target-free, and non-authoritative**. This report is a
diagnostic implementation result, not a frozen scientific benchmark result and
not a tracker-integrable cell result.

## Outcome

On the fixed 2,000-row smoke workload, executing the same eight TabFM ensemble
members across two NVIDIA L4 GPUs increased synchronized model-predict
throughput from **224.98 to 421.99 rows/s**, a **1.8757x speedup** and **93.79%
two-GPU scaling efficiency**. Timed end-to-end batch throughput increased from
224.91 to 421.71 rows/s, a 1.8751x speedup. The serial and parallel arms used
the same member plan and produced byte-identical float32 predictions: maximum
absolute difference 0 and the same prediction-content SHA-256.

This result supports the feasibility of estimator/ensemble parallelism in SDM's
TabFM executor. It does not establish full-validation performance, accuracy, or
the relative merit of context parallelism or data parallelism.

## Setup

| Dimension                  | Diagnostic value                                                                                                                                                                                                                       |
| -------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Dataset and task           | RelBench `rel-hm` / `user-churn`                                                                                                                                                                                                       |
| Representation             | Frozen, target-free, one-row-per-task-example TabFM table materialized from exact two-hop temporal neighborhoods                                                                                                                       |
| Neighborhood policy        | Two hops, fanout `[16,16]`, deterministic `last` temporal sampling                                                                                                                                                                     |
| Feature width              | 132 ordered float32 columns                                                                                                                                                                                                            |
| TRAIN context              | First 1,024 rows of the sealed seed-1 canonical TRAIN selection; TRAIN labels only                                                                                                                                                     |
| Query                      | First 2,000 ordered validation rows; no validation labels available to the process                                                                                                                                                     |
| Batch                      | 250 rows, eight timed batches after one 250-row warmup                                                                                                                                                                                 |
| Ensemble                   | E=8; member RNG policy `member`; model/fit generator seed 0                                                                                                                                                                            |
| Precision                  | CUDA BF16 autocast; resident parameters remained float32; fit caches contained BF16, bool, and int64 tensors                                                                                                                           |
| Compared execution         | `serial_1gpu` on GPU 0 versus `ensemble_parallel_2gpu` on GPUs 0 and 1                                                                                                                                                                 |
| Algorithm revision         | `ensemble-parallel-member-rng-v1`                                                                                                                                                                                                      |
| Placement-invariant plan   | `04a576b67c9b2a6d51cb9a27328af1af8106448316b5be2ea7fffc66d93383f7`                                                                                                                                                                     |
| Host                       | AWS On-Demand `g6.12xlarge`, `i-0760577d3a8d2c2a7`, `us-east-2b`                                                                                                                                                                       |
| GPU inventory              | 4 x NVIDIA L4, 23,034 MiB each by `nvidia-smi`; only one or two GPUs used by the compared arms                                                                                                                                         |
| Driver/runtime             | NVIDIA driver 595.91.07; CPython 3.12.14; PyTorch 2.9.1+cu128                                                                                                                                                                          |
| SDM source                 | Diagnostic-only staged package B v1, 182-member source manifest `34f80621ba09e9f585ab0d6d894169d03b7ef2a4b21e28968815c6b2b6323d83`, based on Git revision `bebaef1ccd074a376dd2f4c3e5c0c010fb8d6db3` plus allowlisted worktree changes |
| Executed diagnostic script | SHA-256 `9467dcf805f993d67a00fe4dfa3e58844c9ffbfca127117b258462ca72bf3189`                                                                                                                                                             |

The requested Spot capacity was not used for this diagnostic. The selected
host was an On-Demand L4 host; this avoids presenting an On-Demand measurement
as Spot evidence. Host market does not change the GPU execution result, but it
does change cost/capacity conclusions.

The staged A3 materialization has 10,000 context rows and 76,556 ordered
validation rows in full. This diagnostic deliberately sliced it to 1,024 and
2,000 rows. Its ten-file, 447,819,926-byte bundle manifest is
`2845ed4d2afea85280fcf5d43d6d6046320f256b6e22ce99b74e2437fc98351a`.
The TabFM classification checkpoint is 6,557,888,408 bytes with SHA-256
`928cb350becdc77cdb7a9e8c36deda88917bfd14a3091894a2dc516db58a2085`.

## Method

The runner first verified the exact A3 inventory and every A3 manifest member,
rejected any validation-target member, and verified the checkpoint size and
hash. Loading the three NumPy slices took 0.0091 s once, before either arm.

Both arms used the same `EnsembleParallel(..., rng_policy="member")` execution
semantics. The one-GPU arm assigned all eight members to GPU 0. The two-GPU arm
assigned even members 0/2/4/6 to GPU 0 and odd members 1/3/5/7 to GPU 1. Each
arm performed these phases independently:

1. construct and load the TabFM replica or replicas;
2. fit/cache the same 1,024-row context with E=8;
3. run one unrecorded 250-row warmup;
4. reset peak CUDA memory and synchronously time all eight 250-row query
   batches, including host-to-device transfer, model predict, and result gather;
5. replay the same eight batches under `torch.profiler` into a separate trace;
6. record resources and close the model.

CUDA synchronization surrounded the measured phases. The primary throughput is
2,000 divided by the sum of the eight synchronized model-predict timings.
End-to-end batch throughput includes the explicitly timed input transfer,
predict, and output gather, but it excludes checkpoint load, fit/cache, warmup,
and profiler replay.

## Timing and throughput

| Metric                                |   Serial, 1 GPU | Ensemble-parallel, 2 GPU |      Parallel / serial effect |
| ------------------------------------- | --------------: | -----------------------: | ----------------------------: |
| Shared preprocessing, before arms     |        0.0091 s |                 0.0091 s |          common, not repeated |
| Cold model load                       |        2.3469 s |                 3.2746 s | 0.7167x speedup; 39.5% slower |
| Context fit/cache                     |        6.9654 s |                 4.8689 s |               1.4306x speedup |
| Warmup predict                        |        1.1350 s |                 0.7583 s |               1.4968x speedup |
| Warmup end-to-end                     |        1.1356 s |                 0.7589 s |               1.4964x speedup |
| Timed model-predict total, 2,000 rows |        8.8898 s |                 4.7394 s |           **1.8757x speedup** |
| Timed end-to-end total, 2,000 rows    |        8.8926 s |                 4.7426 s |           **1.8751x speedup** |
| Model-predict throughput              | 224.9765 rows/s |          421.9908 rows/s |                   **1.8757x** |
| End-to-end throughput                 | 224.9056 rows/s |          421.7109 rows/s |                   **1.8751x** |
| Two-GPU scaling efficiency            |              -- |             **93.7855%** |     model-predict speedup / 2 |
| Mean end-to-end batch time            |        1.1116 s |                 0.5928 s |                 1.8752x lower |
| End-to-end batch p50                  |        1.1119 s |                 0.5930 s |                 1.8752x lower |
| End-to-end batch p95                  |        1.1125 s |                 0.5941 s |                 1.8727x lower |
| End-to-end batch p99                  |        1.1126 s |                 0.5941 s |                 1.8728x lower |
| Mean transfer per batch               |       0.2555 ms |                0.2799 ms |        0.91x; slightly higher |
| Mean result gather per batch          |       0.0894 ms |                0.1086 ms |        0.82x; slightly higher |
| Cold load + fit + warmup + timed e2e  |       19.3405 s |                13.6450 s |               1.4174x speedup |

The eight end-to-end batch measurements were:

| Batch | Serial, 1 GPU (s) | Ensemble-parallel, 2 GPU (s) |
| ----: | ----------------: | ---------------------------: |
|     0 |          1.109607 |                     0.590861 |
|     1 |          1.111074 |                     0.594008 |
|     2 |          1.111192 |                     0.591870 |
|     3 |          1.112625 |                     0.594094 |
|     4 |          1.111990 |                     0.592712 |
|     5 |          1.111888 |                     0.593070 |
|     6 |          1.112007 |                     0.593100 |
|     7 |          1.112235 |                     0.592870 |

## Memory and utilization

| Metric                                 | Serial, 1 GPU | Ensemble-parallel, GPU 0 | Ensemble-parallel, GPU 1 | Two-GPU aggregate |
| -------------------------------------- | ------------: | -----------------------: | -----------------------: | ----------------: |
| Resident parameter bytes               | 6,557,769,832 |            6,557,769,832 |            6,557,769,832 |    13,115,539,664 |
| Fit-cache bytes                        | 3,158,311,896 |            1,579,155,964 |            1,579,155,996 |     3,158,311,960 |
| Peak allocated through fit             |    14.104 GiB |               11.893 GiB |               11.887 GiB |        23.780 GiB |
| Peak reserved through fit              |    14.654 GiB |               13.076 GiB |               12.809 GiB |        25.885 GiB |
| Peak allocated during timed prediction |    13.345 GiB |               11.137 GiB |               11.134 GiB |        22.272 GiB |
| Peak reserved during timed prediction  |    14.654 GiB |               13.076 GiB |               12.809 GiB |        25.885 GiB |
| `nvidia-smi` max used                  |    15,260 MiB |               13,644 MiB |               13,368 MiB |        27,012 MiB |
| Mean sampled GPU utilization           |        60.29% |                   42.26% |                   40.66% |      not additive |
| Maximum sampled GPU utilization        |          100% |                     100% |                     100% |      not additive |

The serial process's maximum host RSS was 7.016 GiB; the parallel process's
was 9.476 GiB. Ensemble parallelism reduced the cache and activation pressure
on each individual GPU, but it replicated the full 6.108 GiB float32 parameter
set on both devices. Consequently, per-GPU peak allocation fell while aggregate
GPU allocation and host RSS increased. This is a throughput tradeoff, not a
memory-saving scheme in aggregate.

Recorded explicit traffic was 2,792,864 host-to-device bytes and 34,000
device-to-host bytes for each arm. The resource hook recorded zero explicit
peer-to-peer bytes. In the parallel arm these controller-facing bytes were
charged to GPU 0; this does not imply that all framework-internal transfers
were absent.

## Profiler findings

The profiler replay categorized events by name and CUDA activity. Values below
are aggregate event self-times from the replay, not wall-clock phase times;
CUDA work can overlap across two devices, so category totals must not be used as
another latency measurement.

| Category        | Serial calls | Serial CPU self (s) | Serial CUDA self (s) | Parallel calls | Parallel CPU self (s) | Parallel CUDA self (s) |
| --------------- | -----------: | ------------------: | -------------------: | -------------: | --------------------: | ---------------------: |
| Compute         |      287,380 |               0.845 |                6.930 |        124,605 |                 0.026 |                  6.833 |
| Coordinator     |       51,914 |               4.402 |                0.000 |        205,929 |                 3.703 |                  0.000 |
| Transfer        |       39,248 |               1.373 |                1.648 |         42,325 |                 1.082 |                  1.861 |
| Synchronization |          409 |               0.002 |                0.000 |          1,073 |                 0.011 |                  0.000 |
| Communication   |            0 |               0.000 |                0.000 |              0 |                 0.000 |                  0.000 |

The traces show the workload remained dominated by TabFM compute, including
BF16 GEMMs, elementwise kernels, reductions, and FlashAttention kernels. There
were no NCCL/all-reduce/all-gather events classified as communication, which is
consistent with independent ensemble members followed by result aggregation
rather than tensor-parallel collective execution. Parallel execution increased
the number of coordinator and synchronization calls, while measured batch
transfer and gather remained below 0.4 ms combined on average. The remaining
approximately 6.2% gap from ideal two-GPU scaling is therefore compatible with
coordinator/synchronization overhead, per-device imbalance, and non-parallel
bookends; this diagnostic does not isolate their individual causal shares.

Mean sampled utilization was lower on each of the two GPUs than on the serial
GPU, even though both parallel GPUs reached 100%. The 0.2-second sampler covered
model load, fit, warmup, timed inference, and profiler replay, including CPU and
idle gaps. These means are useful process-level observations, not a
kernel-window utilization claim.

## Numerical and plan parity

The member plan was exactly equal between arms:

- member-plan digest:
  `24f8f63e34a8ffa206338f3eaa5ba747dee916548e2699f4631929ba595876e3`;
- generator state before planning:
  `b52772af36e654455dc0c2f3bcf63b69ea90c22fd8d4c1d3e4b750afe86d0290`;
- generator state after planning:
  `52153d53a40f303abfb2827edde650dc16374c87ce98481e38d953f70bb0f582`;
- eight member seeds, in estimator order:
  `4798350494943928942`, `1352521532343745542`,
  `1520475452663297742`, `6089310788051350250`,
  `3852547665961953556`, `1588388405525942126`,
  `4365354763458416374`, and `2694194601798398155`.

The saved prediction arrays have the same file SHA-256,
`6b24c81c6939255ac750abbb2fd77830c34d40cd09a84ea5cce45226fa07185f`.
Their contiguous float32 content has SHA-256
`f278e9e4af043e62afc6499d096afe1095754b75f5a9a4a15699603484c691ca`
in both arm records. The direct comparison passed `rtol=5e-3, atol=5e-4` with
maximum absolute difference **0.0**. Thus this run demonstrated exact
prediction parity for the measured smoke slice, not merely tolerance parity.

## Implementation interpretation

The observed behavior matches estimator parallelism's intended boundary:
replicate the TabFM model on each device, partition stable ensemble-member
indices across replicas, keep each member's RNG stream and output position
independent of placement, and aggregate the ordered predictions. The even/odd
cache affinities and unchanged member-plan/prediction hashes provide direct
evidence that placement changed while the logical E=8 computation did not.

For this shape, estimator parallelism is a strong first multi-GPU path because
ensemble members require no cross-device activation exchange. The principal
costs are duplicated checkpoint/model loading, duplicated weights, and modest
coordination. A production implementation should preserve the placement-
invariant member plan, keep serial and parallel arms on the same executor
revision, report cold and warm phases separately, and make device-local query
staging/result aggregation explicit.

This run does not evaluate context parallelism. Context parallelism would split
the 1,024-row in-context state or attention work within an estimator and would
need model-internal sharding plus communication. It also does not evaluate data
parallelism over independent query batches. Data parallelism could be useful
when batches are independent, but each replica would ordinarily retain all E=8
caches, increasing per-device cache duplication unless combined carefully with
ensemble placement. Those alternatives need separate prototypes and parity
tests; the present speedup must not be attributed to them.

## Limitations

- This was explicitly an exploratory diagnostic. It did not have the final
  production package/config/runtime receipt chain and is not reportable science.
- It is one smoke run on one host, one checkpoint, 1,024 context rows, 2,000
  ordered validation rows, eight 250-row batches, and two of four installed L4s.
  There are no repetitions, uncertainty intervals, or full-shape measurements.
- The process had no validation targets and did no scoring. No AUROC, accuracy,
  or RelBench evaluator was computed. The result says nothing about predictive
  quality.
- The materialized two-hop table is a fixed TabFM input representation, not an
  endorsed RelBench model and not equivalent to native relational execution.
- Cold checkpoint/model load was measured but not optimized. The two-GPU arm
  loaded a complete checkpoint/model replica per GPU and took longer to load.
- Profiler replay occurred after the timed pass and incurs profiler overhead.
  Category rules are coarse, and summed CUDA self-time across devices is not
  wall time.
- No L40S, Spot, 3/4-GPU, context-parallel, data-parallel, KumoRelational, or
  cross-host result was measured.
- The exact output equality applies to this smoke input and software stack; it
  is not a general determinism guarantee across hardware/runtime changes.

## Artifact and integrity evidence

Sealed host attempt root:

```text
/opt/dlami/nvme/sdm-bench/attempts/diagnostic-exploratory-v4-20260827T050734Z-649988af
```

Controller-synced evidence root:

```text
/Users/ardrianw/repositories/.benchmark-artifacts/sdm-tabfm-mg-l4-od-20260827T043507Z/diagnostic-exploratory-v4-20260827T050734Z-649988af
```

Key immutable members:

| Artifact                            | SHA-256                                                            |
| ----------------------------------- | ------------------------------------------------------------------ |
| `SHA256SUMS`                        | `512366fd01443dbe14c3dbaf2c89946b499c1f744cc75aaf50f627aac96de17f` |
| `observed/diagnostic-complete.json` | `d1187440ab77f51df938a3d72bbba647b326a9ea6b67127a00f2d4b5f6c5fc21` |
| `observed/serial.json`              | `b4fc8e6fd275c798a35c94b56c8b0983dd1ceaa7c38dec8636ac6364613b159e` |
| `observed/parallel.json`            | `d59fa3fe3b4fceb8205fd4a77553ce2c127f8f7a428d9cffdcdb065f52da747d` |
| Serial profiler trace               | `aa75d4e96c40c1da2339f47ab9dde3a25c5b8b23a9abc2319f49ba09250dacb2` |
| Parallel profiler trace             | `5b566c384f57a05e9b972d152f1681538d1c3afb64d36e4ddb6a8de54fce92d0` |
| Each saved prediction array         | `6b24c81c6939255ac750abbb2fd77830c34d40cd09a84ea5cce45226fa07185f` |
| Executed `diagnostic.py`            | `9467dcf805f993d67a00fe4dfa3e58844c9ffbfca127117b258462ca72bf3189` |

The controller independently recomputed every member listed in the sealed
`SHA256SUMS`; all 12 members passed, and the manifest itself recomputed to the
digest above. It also recomputed the final receipt, serial record, and parallel
record hashes and found that they matched the receipt's internal bindings. This
is independent transport/inventory and binding verification, not an independent
scientific recomputation of timing or speedup and not an independent semantic
validation of the diagnostic runner.

The completion event is recorded as event
`1f78e050-b18a-4cce-b763-f59610fb93f6` in the controller host journal at:

```text
/Users/ardrianw/repositories/.benchmark-artifacts/sdm-tabfm-mg-l4-od-20260827T043507Z/host-operations.jsonl
```
