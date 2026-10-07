# Table operation mode restoration under compilation

This branch builds on `compile/atomic-table-dispatch`. It adds a narrow workaround for a PyTorch 2.7 Dynamo resumption failure; it does not change table data or model arithmetic.

## Failure and change

In a real KumoRelational `fit()` run with graph breaks allowed, Dynamo resumed inside the wrapper that restores inference/autograd state and raised `InternalTorchDynamoError: KeyError: 276` at `return fn(*args, **kwargs)`. The failure reproduced with fresh caches and both dynamic-shape settings.

The wrapper now has `@torch.compiler.disable(recursive=False)`. Python mode setup and cleanup stay together, while called tensor handlers remain eligible for compilation. The existing inference-mode and gradient-mode rules are unchanged. This is not a boundary around all fitting, preprocessing, or neural computation.

After that boundary, tracing reached the concatenation handler and failed resuming a bound `ref.items()` generator. Its two loops now call `TableTensor.items(ref)` directly, so Dynamo inlines ordinary Python iteration instead of treating a tensor method returning a generator as a tensor operation. The semantic-type order and selected blocks are unchanged.

## Validation

Actual CPU Inductor on PyTorch 2.7.1 and 2.14, both graph policies:

- Row and column concatenation pass for 3, 5 and 0 rows, including noncontiguous numerical data and datetime blocks.
- Existing table and container-compilation checks report 60 passed / 8 CUDA skips on 2.14 and 58 passed / 8 skips on 2.7; its two known main pandas default-device failures are excluded.
- The real 2.7 fitting diagnostic advances past the mode-wrapper exception. This does not establish complete public `fit()` or fullgraph model support; the integration investigation tracks subsequent failures.

No speed or memory improvement is claimed. This branch adds no dtype conversion, new random draw, or numerical fallback.
