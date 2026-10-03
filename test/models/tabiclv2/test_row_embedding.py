from collections.abc import Iterator

import pytest
import torch

from sdm.cache import Cache
from sdm.models.tabiclv2.row_embedding import RowEmbedding
from sdm.testing import withCUDA


@pytest.fixture
def reset_compiler() -> Iterator[None]:
    torch.compiler.reset()
    yield
    torch.compiler.reset()


@withCUDA
@pytest.mark.usefixtures("reset_compiler")
@pytest.mark.parametrize("num_classes", [0, 3, 13])
@pytest.mark.parametrize("batch_shape", [(), (2, 1)])
def test_row_embedding_compiled_replay(
    device: torch.device,
    num_classes: int,
    batch_shape: tuple[int, ...],
) -> None:
    encoder = RowEmbedding(
        num_classes=4 if num_classes else 0,
        channels=16,
        num_layers=2,
        num_heads=2,
        group_size=3,
        num_inducing_points=4,
        num_readout_tokens=2,
        norm_bias=False,
        device=device,
    ).eval()

    with torch.inference_mode():
        # Zero-initialized residual projections would hide lost column updates.
        for module in encoder.modules():
            if isinstance(module, torch.nn.Linear):
                module.weight.normal_(std=0.1)
                if module.bias is not None:
                    module.bias.normal_(std=0.02)

        context = torch.randn(*batch_shape, 11, 5, device=device)
        labels = (
            torch.randint(num_classes, (*batch_shape, 11), device=device)
            if num_classes
            else torch.randn(*batch_shape, 11, device=device)
        )
        cache = Cache()
        encoder(context, labels, num_classes=num_classes, cache=cache)
        cache = cache.freeze()
        query = torch.randn(*batch_shape, 3, 5, device=device)

        expected = encoder(
            query,
            labels[..., :0],
            num_classes=num_classes,
            cache=cache,
        )
        compiled = torch.compile(
            encoder,
            backend="eager",
            fullgraph=True,
            dynamic=False,
        )
        actual = compiled(
            query,
            labels[..., :0],
            num_classes=num_classes,
            cache=cache,
        )

        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
