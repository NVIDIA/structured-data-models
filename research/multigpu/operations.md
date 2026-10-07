# EC2 Spot operations

Task: `kumo-multigpu-20261008`. The cloud operator owns resource mutations. Detailed commands and machine-readable receipts live outside the repository in `/Users/ardrianw/repositories/.kumo-multigpu-20261008/ops`.

Initial capacity plan: one Spot host with four GPUs, preferring L40S (`g6e.12xlarge`), then L4 (`g6.12xlarge`) or A10G (`g5.12xlarge`), according to observed capacity and price. Initial spend envelope is $100. Each launched host receives an eight-hour operating-system shutdown timer and EC2 shutdown-to-terminate behavior; root volumes are encrypted and deleted on termination. Additional hosts require coordinator scheduling within the user's authorized investigation.

No On-Demand fallback is authorized. Existing machines and resources outside this task are preserved. Scientific runners share one host via explicit GPU reservations.

## Authentication

Initial local AWS identity check on 2026-10-08 failed because the AWS SSO session expired. The existing SSH controller also uses this local token for its SSM proxy. Device-code authentication has been initiated; no instances were launched while waiting for login.

Authentication succeeded via the user's `al` shell function (ordinary browser-based `aws sso login`).

## Active host

| Field | Observed value |
|---|---|
| Instance | `i-0f247930321bbf67e` |
| Market/type | Spot, `g6e.12xlarge` |
| Region/zone | `us-east-2` / `us-east-2a` |
| GPUs | 4 × NVIDIA L40S, 48 GB each |
| Initial compute quote | $5.2987/hour |
| Launch | 2026-10-07 15:16:48 UTC |
| Shutdown deadline | 2026-10-07 23:15:34 UTC, instance shutdown behavior `terminate` |
| Root disk | 200 GB encrypted gp3, delete on termination |
| Operating system | Ubuntu 24.04, October 3 PyTorch 2.9 DLAMI |
| Runtime | Python 3.12, PyTorch 2.9.1+cu130, pyg-lib 0.7.0+pt29cu130 |
| Topology | All GPU pairs `NODE`: PCIe host bridges within one NUMA node; no NVLink |

The initial ten eligible four-GPU pools in us-west-2 returned insufficient Spot capacity: three L40S pools, four L4 pools, and three A10G pools. Switching to us-east-2 secured L40S capacity on the first attempt. Cloud initialization took approximately 171 seconds; initial PyTorch import stalled on cold EBS reads before succeeding. Runtime dependencies are installed and package versions saved in `ops/host-runtime/pip-freeze.txt`.

A separate proposed `p4d.24xlarge` eight-A100 NVSwitch Spot experiment had an attractive $4.7798/hour quote but was explicitly denied by an AWS organization Service Control Policy (`p-aht2tzt8`). No instance was launched. This topology remains unmeasured; the denied instance family is not retried.

Versioned source directories preserve each committed source snapshot. The initial snapshot is `713489ca6`; subsequent harness/implementation updates use additional directories. Both pretrained families use portable Hugging Face cache directories populated from exact revisions, with offline model loading. Real-data files contain TRAIN and validation only.

## Second host and scale constraints

A second four-L40S host in us-east-2 was denied with `MaxSpotInstanceCountExceeded`. Read-only quota inspection showed the regional **All G and VT Spot Instance Requests** quota is 64 vCPUs; the first `g6e.12xlarge` uses 48. The same 64-vCPU quota in us-east-1 permits one additional four-GPU host there. Eight-GPU G-family shapes require 192 vCPUs and therefore cannot be launched within these existing quotas. No quota-increase request was made.

After insufficient-capacity responses for the affordable L40S and lower-priced L4 pools in us-east-1, a second host launched:

| Field | Observed value |
|---|---|
| Instance | `i-05cea203a4e1895fd` |
| Market/type | Spot, `g6.12xlarge` |
| Region/zone | `us-east-1` / `us-east-1d` |
| GPUs | 4 × NVIDIA L4, 24 GB each |
| Initial compute quote | $4.0969/hour |
| Launch | 2026-10-07 15:28:23 UTC |
| Verified shutdown deadline | 2026-10-07 19:28:00 UTC, shutdown-to-terminate |
| Assigned experiments | Context parallelism, attention-kernel variants, stage/layer placement |
| Initial source snapshot | `55dba6bb5` |

The L40S host runs native/ensemble/data-parallel comparisons; the L4 host runs its own native/context/placement comparisons. Speedup denominators must come from the same host and GPU model. The public checkpoint revisions and input artifacts are shared. The initial combined compute rate is $9.3956/hour; the two different shutdown timers bound the initial total below $60 at these quotes.

Nsight Systems CLI 2026.5.1.161 was installed on the L40S host during a pause between timed runs, following NVIDIA's [installation guide](https://docs.nvidia.com/nsight-systems/InstallationGuide/index.html). The same installation is included in the second host setup. Nsight captures GPU streams from worker threads that may be missing from a coordinator-only PyTorch profile; its instrumented timings are kept separate from clean throughput measurements.

The second host completed installation and direct public checkpoint downloads. All 15 manifest-listed model files were independently rehashed on that host and matched their expected SHA-256 and size. Its runtime matches the first host: Python 3.12, PyTorch 2.9.1+cu130, pyg-lib 0.7.0+pt29cu130, and Nsight Systems CLI 2026.5.1. All L4 GPU pairs also report `NODE` topology. The prepared relational query/context graph artifacts were copied directly from the first host, preserving identical samples.

## Cleanup tracking

The unused us-west-2 task security group `sg-0e1cf114cb4fb68cb` and imported task key pair `key-0cc808fae3cc35d69` were removed after a read-only check confirmed that no task instances existed in that region. This removed only empty temporary cloud access resources; the user's local SSH key remains unchanged. Active host resources in us-east-1 and us-east-2 remain until evidence collection and experiment completion.

## Evidence

Provisioning records will include current Spot quotes, selected availability zone, instance and volume identifiers, GPU topology, driver/runtime versions, capacity failures, and teardown status. Benchmarks must record the exact source revision, command, workload, GPU selection, synchronization-aware timing, peak memory, and prediction comparison.
