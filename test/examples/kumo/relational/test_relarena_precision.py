"""The benchmark's numerical policy is explicit and scoped."""

import pytest
import torch
from examples.kumo.relational._relarena.adapter import v11_precision


def test_v11_backend_precision_restores_after_failure() -> None:
    before = (
        torch.get_float32_matmul_precision(),
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
        torch.backends.cuda.cudnn_sdp_enabled(),
    )

    def fail_inside_precision() -> None:
        with v11_precision():
            assert torch.get_float32_matmul_precision() == "highest"
            assert not torch.backends.cuda.matmul.allow_tf32
            assert not torch.backends.cudnn.allow_tf32
            assert not torch.backends.cuda.cudnn_sdp_enabled()
            raise RuntimeError("probe")

    with pytest.raises(RuntimeError, match="probe"):
        fail_inside_precision()
    assert before == (
        torch.get_float32_matmul_precision(),
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
        torch.backends.cuda.cudnn_sdp_enabled(),
    )
