from collections.abc import Callable

import torch
from torch import Tensor
from torch.nn import RMSNorm

# Small tensors do not amortize the custom launch overhead.
_MIN_ROWS = 1024

_rmsnorm_cast: Callable[[Tensor, Tensor, float], Tensor] | None = None
try:
    from sdm._kernels.triton.rmsnorm_cast import rmsnorm_cast
except ImportError:
    pass
else:
    _rmsnorm_cast = rmsnorm_cast


class _RMSNormForLinear(RMSNorm):
    """RMSNorm whose output is consumed only by an autocast Linear."""

    def forward(self, x: Tensor) -> Tensor:
        if torch.compiler.is_compiling() or x.numel() < _MIN_ROWS * x.size(-1):
            return super().forward(x)
        if (
            _rmsnorm_cast is not None
            and not self.training
            and not torch.is_grad_enabled()
            and x.is_cuda
            and torch.version.hip is None
            and torch.is_autocast_enabled("cuda")
            and x.dtype in {torch.float16, torch.bfloat16, torch.float32}
            and x.size(-1) in {128, 256, 512}
            and self.weight is not None
            and self.weight.dtype == torch.float32
            and self.weight.device == x.device
            and self.weight.is_contiguous()
        ):
            return _rmsnorm_cast(
                x,
                self.weight,
                torch.finfo(torch.float32).eps
                if self.eps is None
                else self.eps,
            )
        return super().forward(x)
