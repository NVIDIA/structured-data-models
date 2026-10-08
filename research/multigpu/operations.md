# EC2 Spot operations

Task: `kumo-multigpu-20261008`. The cloud operator owns resource mutations. Detailed commands and machine-readable receipts live outside the repository in `/Users/ardrianw/repositories/.kumo-multigpu-20261008/ops`.

Initial capacity plan: one Spot host with four GPUs, preferring L40S (`g6e.12xlarge`), then L4 (`g6.12xlarge`) or A10G (`g5.12xlarge`), according to observed capacity and price. Initial spend envelope is $100. Each launched host receives an eight-hour operating-system shutdown timer and EC2 shutdown-to-terminate behavior; root volumes are encrypted and deleted on termination. Additional hosts require coordinator scheduling within the user's authorized investigation.

No On-Demand fallback is authorized. Existing machines and resources outside this task are preserved. Scientific runners share one host via explicit GPU reservations.

## Authentication

Initial local AWS identity check on 2026-10-08 failed because the AWS SSO session expired. The existing SSH controller also uses this local token for its SSM proxy. Device-code authentication has been initiated; no instances were launched while waiting for login.

Authentication succeeded via the user's `al` shell function (ordinary browser-based `aws sso login`).

## Initial host (historical)

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

The same 15-file model rehash subsequently passed on the first host during a pause between timed experiment queues. Both machines therefore use verified identical pretrained checkpoint bytes.

## Attempted eight-L4 distributed data parallelism

The coordinator authorized four additional `g6.xlarge` Spot workers (one L4 and four vCPUs each), which would fit the remaining 16 vCPUs of the regional quota beside the four-GPU L4 host. This would provide eight homogeneous L4 GPUs across hosts, while explicitly retaining heterogeneous CPU/network resources. The proposed transport was persistent SSH workers with a task-only key generated on the L4 coordinator; its private key never left that host. No public service listener was planned.

Same-zone `us-east-1d` attempts failed both with an exact count of four and a flexible count of one through four. All remaining eligible zones in us-east-1 (`1a`, `1b`, `1c`, then `1f` with a raised $3/hour combined worker ceiling) also returned insufficient Spot capacity. Cross-region fallback attempts in all three us-east-2 zones failed identically. No additional worker instance was launched and no worker compute charge accrued. West-region quotes were inspected, but no further launch was attempted after the coordinator's bounded search window.

Homogeneous eight-L4 scaling is therefore an unmeasured capacity limitation of this experiment, not a measured software limitation. The reproducible implementations and worker orchestration can be rerun when capacity becomes available. Four-GPU measurements on each original host remain the primary empirical scaling evidence.

The coordinator subsequently authorized a separate heterogeneous eight-GPU query-DP arm using the two existing machines: four L4s in us-east-1d and four L40S GPUs in us-east-2a. This adds no capacity or compute resources. Its results must be labeled cross-region and heterogeneous, with matched four-L4 and four-L40S baselines rather than a homogeneous eight-GPU speedup claim. The task-only SSH key on the L4 coordinator was authorized on the L40S host; its security group permits SSH solely from the coordinator's public `/32` address. No service port was opened. The now-unused same-region private SSH rule was revoked. Both the cross-region SSH ingress rule and the task identity are removed with task resource teardown.

## Cleanup tracking

The unused us-west-2 task security group `sg-0e1cf114cb4fb68cb` and imported task key pair `key-0cc808fae3cc35d69` were removed after a read-only check confirmed that no task instances existed in that region. This removed only empty temporary cloud access resources; the user's local SSH key remains unchanged. Active host resources in us-east-1 and us-east-2 remain until evidence collection and experiment completion.

## Access restriction during continuation

At 2026-10-07 16:30:55 UTC the execution environment changed to workspace-write filesystem access, restricted networking, and approval policy `never`. Fresh read-only SSH attempts to both task hosts failed immediately with `Operation not permitted`; read-only AWS instance queries failed to connect to both regional EC2 endpoints. No alternate access route, new instance, restart, or permission bypass was attempted.

The last successful lifecycle probes, at 16:27:25 UTC, observed both hosts reachable and no Spot interruption notice. Current process state, any newly completed remote result files, and termination status are **unverified** after the restriction. The relational owner last reported a five-arm large-context queue under `results/relational-hm-c16384-b512-*`; these arms must not be marked complete or rerun until actual host state can be inspected.

The previously verified shutdown-to-terminate timers remain the fallback: L4 host at 19:28:00 UTC and L40S host at 23:15:34 UTC. Their subsequent execution is not asserted without a successful EC2 state query. After network access is restored, retrieve and checksum remaining results, confirm scientific workers are finished, terminate both task instances, then remove the remaining task access resources:

| Region | Instance | Security group | Imported key-pair name |
|---|---|---|---|
| us-east-1 | `i-05cea203a4e1895fd` | `sg-03014ad8f6269f6ec` | `kumo-multigpu-20261008` |
| us-east-2 | `i-0f247930321bbf67e` | `sg-01bcb7f6ec49e2c71` | `kumo-multigpu-20261008` |

The cross-region SSH rule `sgr-028d1b8e6657ba968` belongs to the us-east-2 task security group. The earlier private SSH rule `sgr-03b0ff34c5bc7c929` was already revoked. The task-only private SSH identity exists solely on the L4 host's encrypted, delete-on-termination root volume. No current cloud resource has been claimed terminated merely because direct access is blocked.

## Evidence

Provisioning records will include current Spot quotes, selected availability zone, instance and volume identifiers, GPU topology, driver/runtime versions, capacity failures, and teardown status. Benchmarks must record the exact source revision, command, workload, GPU selection, synchronization-aware timing, peak memory, and prediction comparison.

## Restored access and replacement capacity (2026-10-08)

Network access was restored at approximately 01:34 UTC, after both original shutdown deadlines. Read-only EC2 checks found neither original instance and no task-tagged EBS volumes in either region. The Ohio Spot request was closed with `instance-terminated-by-user` and update time 2026-10-07 23:23:31 UTC. The Virginia request had already been purged, so its exact termination time/cause cannot be independently reconstructed. Remote-only evidence from the interrupted queues is not recoverable from surviving task disks; previously downloaded evidence remains valid. The authoritative receipt is `ops/old-host-lifecycle-resume-20261008.json`. The obsolete cross-region SSH ingress rule was revoked.

Two fresh Spot hosts were acquired for explicitly new attempts:

| Role | Instance / type | Zone | GPUs | Launch UTC | Compute quote | Shutdown guard |
|---|---|---|---|---|---|---|
| Pretrained EP and relational controls | `i-0fa5c18a88d2eb124`, `g6e.12xlarge` | us-east-2a | 4 L40S | 01:38:13 | $5.3029/hour | 04:38 UTC |
| CP/GNN gates, then joint query DP | `i-03ea6eecfa001b1a6`, `g6.12xlarge` | eu-central-1b | 4 L4 | 01:49:47 | $2.7675/hour | 03:50 UTC |

All eligible Virginia L40S/L4/A10 and Oregon L40S/L4/A10 fallback pools returned insufficient capacity. Frankfurt L4 capacity succeeded within the existing 64-vCPU quota. No denied P-family launch was retried, no quota increase was requested, and no On-Demand instance was used. Any joint eight-GPU result is therefore heterogeneous and cross-Atlantic, not homogeneous single-host scaling. Both hosts use encrypted 200 GB gp3 root volumes with delete-on-termination and shutdown-to-terminate behavior. Combined quoted compute rate is $8.0704/hour; the original $100 was an initial planning envelope, not a user-imposed cap.

Both replacement runtimes are verified as PyTorch 2.9.1+cu130 with pyg-lib 0.7.0+pt29cu130; all 15 checkpoint-manifest files passed SHA-256 verification on each host. Immutable `source-c0fb64fcc` and exact prepared data/graphs were independently verified before handoff. Later source revisions use separate directories. Each runner must download evidence after individual arms rather than defer recovery until a full queue ends. Replacement runtime receipts are stored separately under `ops/resume-host1-runtime` and `ops/resume-host2-runtime`.

After confirming no live task instances remained in Virginia or Oregon, their unused task security groups and imported task key pairs were deleted. Only Ohio and Frankfurt access resources remain. A fresh task-only SSH key was generated on the Frankfurt coordinator; its private key remains there, and only its public key is authorized on the Ohio task host. Ohio ingress rule `sgr-01052243ff6022566` allows TCP 22 exclusively from the Frankfurt host's public `/32`; this rule and both active regional access resources must be removed at final teardown. The local user SSH key and unrelated controller remain untouched. Access-resource receipts are in `ops/resume-access-resources.json`.
