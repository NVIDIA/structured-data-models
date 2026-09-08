# ruff: noqa: D101, D102, A002

import torch
from torch import Tensor


class Linear(torch.nn.Linear):
    def forward(
        self,
        input: Tensor,
        *,
        out: Tensor | None = None,
    ) -> Tensor:
        if out is None:
            return super().forward(input)

        if torch.is_grad_enabled():
            raise RuntimeError(
                "'out' is only supported when gradients are disabled"
            )

        torch.matmul(
            input.to(out.dtype),
            self.weight.to(out.dtype).t(),
            out=out,
        )

        if self.bias is not None:
            out += self.bias.to(out.dtype)

        return out
