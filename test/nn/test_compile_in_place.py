import copy
from collections.abc import Iterator

import pytest
import torch
from torch import Tensor

from sdm.cache import KVCacheEntry
from sdm.nn import InducedTransformerBlock, QASSMax, TransformerBlock
from sdm.testing import withCUDA


@pytest.fixture(autouse=True)
def _reset_dynamo() -> Iterator[None]:
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


@withCUDA
@pytest.mark.parametrize("induced", [False, True])
@pytest.mark.parametrize("grad_mode", ["inference_mode", "no_grad"])
def test_transformer_block_compile_in_place(
    device: torch.device,
    induced: bool,
    grad_mode: str,
) -> None:
    # `TabICLv2.predict` runs its column and row blocks in place on views of
    # one pre-allocated buffer under `torch.inference_mode()` (or under
    # `torch.no_grad()` when a callback requires gradients), so `out`
    # aliases `query` and `key_value`. Compiling the blocks in place makes
    # them separate graph inputs that AOTAutograd has to merge, which used to
    # fail for inference tensors (they carry no `_base`) and, with a
    # different error, for `no_grad` views of a shared storage.
    # Only the in-place `module.compile()` form is covered: the wrapper form
    # `torch.compile(block)(..., out=...)` bypasses `Module.__call__` and
    # keeps failing like it did before the fix.
    # Seed the module init and the buffers below so a numerics failure
    # reproduces from a fixed model and input.
    torch.manual_seed(0)
    channels, K, R_train, C = 8, 2, 3, 4

    def block(qassmax: bool) -> TransformerBlock:
        b = TransformerBlock(
            channels=channels,
            num_query_heads=2,
            mlp=torch.nn.Sequential(
                torch.nn.Linear(channels, 2 * channels, device=device),
                torch.nn.GELU(),
                torch.nn.Linear(2 * channels, channels, device=device),
            ),
            query_scaling=QASSMax(channels // 2, num_heads=2, device=device)
            if qassmax
            else None,
            device=device,
        )
        # `Attention` zero-initialises its output projection, which would
        # make the block ignore `key_value` and hide read-after-write bugs on
        # the aliased buffer. The MLP is a real (randomly initialised) one for
        # the same reason: with `Identity` both blocks would reduce to
        # `2 * query` and the numerics check below would be vacuous.
        with torch.no_grad():
            torch.nn.init.normal_(b.attn.out_lin.weight, std=0.5)
            torch.nn.init.normal_(b.attn.out_lin.bias, std=0.5)
        return b

    module: torch.nn.Module = block(qassmax=False)
    if induced:
        module = InducedTransformerBlock(
            channels=channels,
            num_inducing_points=4,
            inducing_block=block(qassmax=True),
            output_block=module,
            device=device,
        )
    reference = copy.deepcopy(module)
    # The `aot_eager` backend runs AOTAutograd's input aliasing analysis
    # without code generation.
    module.compile(fullgraph=True, dynamic=True, backend="aot_eager")

    def run(block: torch.nn.Module, buffer: Tensor) -> KVCacheEntry:
        # Column pass: non-contiguous transposed views, `out` is `query`.
        col = buffer[..., K:, :].transpose(-2, -3)  # [C, R, D]
        out = block(
            query=col,
            key_value=col[..., :R_train, :],
            batch_size_limit="auto",
            out=col,
        )
        assert out is col

        # Row pass: contiguous `out` that is a sub-view of `key_value`.
        row = buffer[..., :K, :]  # [R, K, D]
        out = block(
            query=row,
            key_value=buffer,
            batch_size_limit="auto",
            out=row,
        )
        assert out is row

        # Cache-recording pass (what `fit()` does): the block returns a tuple
        # while `out` still aliases `key_value`.
        out, key_value = block(
            query=row,
            key_value=buffer,
            return_key_value=True,
            batch_size_limit="auto",
            out=row,
        )
        assert out is row
        assert isinstance(key_value, KVCacheEntry)
        return key_value

    grad_context = (
        torch.inference_mode()
        if grad_mode == "inference_mode"
        else torch.no_grad()
    )
    with grad_context:
        for R in [6, 9]:  # Two shapes exercise the dynamic path.
            buffer = torch.randn(R, K + C, channels, device=device)
            expected = buffer.clone()
            key_value = run(module, buffer)
            expected_key_value = run(reference, expected)
            # The whole pre-allocated buffer is filled in place.
            assert torch.is_inference(buffer) == (
                grad_mode == "inference_mode"
            )
            torch.testing.assert_close(buffer, expected)
            torch.testing.assert_close(key_value.key, expected_key_value.key)
            torch.testing.assert_close(
                key_value.value, expected_key_value.value
            )

    # The compiled block keeps rejecting `out` when gradients are enabled.
    buffer = torch.randn(6, K + C, channels, device=device)
    with pytest.raises(RuntimeError, match="gradients are disabled"):
        module(query=buffer, key_value=buffer, out=buffer)
