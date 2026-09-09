# ruff: noqa: D101, D102, A001, A002

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

        input = input.to(out.dtype)
        weight = self.weight.to(out.dtype).t()

        if torch.compiler.is_compiling() and not out.is_contiguous():
            out.copy_(torch.matmul(input, weight))
        elif input.dim() == 2:
            torch.matmul(input, weight, out=out)
        else:
            input = input.view(-1, input.size(-2), input.size(-1))
            torch.bmm(
                input,
                weight.expand(input.size(0), -1, -1),
                out=out.view(-1, out.size(-2), out.size(-1)),
            )

        if self.bias is not None:
            out += self.bias.to(out.dtype)

        return out
