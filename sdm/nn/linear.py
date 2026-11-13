# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: D101, D102, A001, A002

import math

import torch
import torch.nn.functional as F
from torch import Tensor


def _linear(
    input: Tensor, weight: Tensor, bias: Tensor | None = None
) -> Tensor:
    if (
        not torch.compiler.is_dynamo_compiling()
        or input.dim() < 3
        or not input.is_contiguous()
    ):
        return F.linear(input, weight, bias)

    # The matrix product reads the input as [N, C]. Compiled code copies the
    # input unless it can prove that this is a view, which fails for leading
    # sizes that are sums, such as `num_columns + 4`. Strides state the view.
    *size, channels = input.size()
    input = input.as_strided((math.prod(size), channels), (channels, 1))
    return F.linear(input, weight, bias).unflatten(0, size)


class Linear(torch.nn.Linear):
    def forward(
        self,
        input: Tensor,
        *,
        out: Tensor | None = None,
    ) -> Tensor:
        if out is None:
            return _linear(input, self.weight, self.bias)

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
            batch_size = math.prod(input.shape[:-2])
            input = input.view(batch_size, input.size(-2), input.size(-1))
            torch.bmm(
                input,
                weight.expand(input.size(0), -1, -1),
                out=out.view(batch_size, out.size(-2), out.size(-1)),
            )

        if self.bias is not None:
            out += self.bias.to(out.dtype)

        return out
