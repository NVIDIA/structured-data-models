# Relational prediction precision localization

Source before the candidate: `c75ba3af6`. Candidate: [`2ce3b6bcc`](https://github.com/NVIDIA/structured-data-models/commit/2ce3b6bccdc1ff24a0b093da8923581be05c852c) on `compile/preserve-linear-bmm`. This experiment adds no production changes.

The real RelBench driver-dnf bundle, arm 1/query 4, contains 16 query rows and six related tables. With the pretrained relational classifier, CPU FP32, one estimator, and the same eagerly fitted model, compiling only the internal model with `backend="eager", fullgraph=True` changes final class probabilities by up to 6.267428e-5 and fails the existing tolerance. This backend traces the model but does not run Inductor optimization.

Forward hooks locate the first difference at `row_embedding.lin`. Its output already differs by 1.19e-7–2.38e-7 across the six tables. Differences propagate through row embedding, GNN, and prediction head. This relational model uses the TabICLv2 row embedding with a linear projection; it does not use the Fourier cell embedding discussed in #1054.

The cause is a compilation-specific branch in `sdm.nn.Linear`: eager batched inference uses `torch.bmm(..., out=buffer)`, while the compiled strided-buffer fallback uses `torch.matmul(...)` followed by a copy. These operations express the same mathematics but select different native matrix multiplication shapes/kernels. Capturing the six real inputs establishes:

- Native matmul-plus-copy exactly reproduces the traced compilation output.
- Functional bmm-plus-copy is bitwise equal to eager bmm-with-output-buffer.
- The candidate keeps bmm for batched compiled inputs. Eager behavior and two-dimensional compiled inputs are unchanged.

With the candidate, all 14 recorded neural outputs and final probabilities become bitwise equal under backend="eager" on the 16-row case. BF16 component checks on all six captured shapes also match eager exactly under backend="eager" on PyTorch 2.7.1 and 2.14; this is component validation, not full-model BF16 or GPU validation.

Actual Inductor still introduces later differences. A hooked 2.14 diagnostic has bitwise-equal initial Linear outputs after the candidate but differences later inside row embedding. Therefore the candidate fixes an avoidable operation change; it does not establish universal model parity. In particular, initial unhooked 2.14 checks still find a strict tolerance miss on query 3. No tolerance was relaxed, and no GPU performance or memory claim is made.

## Reproduce

`localize.py` loads the existing trusted bundle and checkpoint, fits eagerly, then compares hooked eager and internally compiled prediction. It writes layer comparisons to JSON and captured Linear inputs to a local `.pt` file. Select the baseline or candidate source through `PYTHONPATH`.

```sh
OMP_NUM_THREADS=1 PYTHONPATH=/path/to/selected/source python localize.py \
  --data /path/to/driver-dnf_bundle.pt \
  --checkpoint /path/to/relational-classifier.pt \
  --output /tmp/layers.json

OMP_NUM_THREADS=1 PYTHONPATH=/path/to/selected/source python linear.py \
  --inputs /tmp/layers.pt --dtype float32
```

Use `--dtype bfloat16` for the native-operation component comparison. Use `--backend inductor` with the layer probe for a code-generation diagnostic, but note that hooks retain intermediate tensors and can affect fusion. The separate unhooked public prediction probe is used to validate final model predictions with actual Inductor.

## Unhooked Inductor results

The candidate executes all four tested cases on both CPU runtimes. These use fullgraph, dynamic shapes, one eager fit followed by internal-module compilation, isolated compiler cache directories, and disabled compiler caches. The table reports the unchanged per-value tolerance, not just maximum error.

| Query case | Rows | PyTorch 2.7.1 | PyTorch 2.14 |
|---|---:|---|---|
|0|1|Pass, max 2.56e-6|Pass, max 3.07e-6|
|3|16|Pass, max 3.99e-5|Fail, max 3.10e-5|
|4|16|Pass, max 3.76e-5|Pass, max 5.19e-5|
|5|16|Fail, max 3.65e-5|Pass, max 2.99e-5|

The candidate changes which Inductor samples miss tolerance; it is not uniformly better on every final probability. For example, 2.14 query 3 includes eager 0.1650883108 versus compiled 0.1651193500: difference 3.1039e-5 exceeds the allowed 2.6509e-5. The operation-preservation fix and the remaining generated-arithmetic issue must be assessed separately. Existing Linear and relational model suites pass 15 CPU cases with 13 CUDA skips on each runtime. GPU performance/memory validation is separate and was not performed by these scripts.
