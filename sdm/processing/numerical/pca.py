# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from typing import Literal, cast

import torch
from torch import Tensor

from sdm import Stype, TableTensor
from sdm.processing import Processor


class PCA(Processor):
    r"""Project numerical columns onto their principal components.

    Args:
        num_components: Number of principal components to keep.
        solver: PCA solver to use: ``"full"`` computes full SVD, ``"eigh"``
            decomposes the feature covariance matrix, and ``"qr"`` or ``"lu"``
            use randomized SVD with QR or LU normalization. ``"auto"``
            selects a solver based on the input shape and number of components.
        num_iterations: Number of power iterations for randomized SVD.
        num_oversamples: Number of additional random vectors used to
            approximate the principal subspace.
    """

    handles_stypes = (Stype.numerical,)
    requires_fit = True

    mean: Tensor
    components: Tensor

    def __init__(
        self,
        num_components: int,
        solver: Literal["auto", "full", "eigh", "qr", "lu"] = "auto",
        num_iterations: int = 7,
        num_oversamples: int = 10,
    ) -> None:
        super().__init__()
        self.num_components = num_components
        self.solver = solver
        self.num_iterations = num_iterations
        self.num_oversamples = num_oversamples

        self.register_buffer("mean", torch.empty(0))
        self.register_buffer("components", torch.empty(0))

    def _fit(
        self,
        table: TableTensor,
        *,
        generator: torch.Generator | None = None,
    ) -> None:

        x = table.numerical
        self._set_buffer("mean", x.mean(dim=-2, keepdim=True))
        x = x - self.mean

        *_, R, C = x.size()
        num_components = min(R, C, self.num_components)

        solver = self.solver
        if solver == "auto":
            if C <= 1024 and R >= 10 * C:
                solver = "eigh"
            elif max(R, C) <= 500:
                solver = "full"
            elif num_components < 0.8 * min(R, C):
                solver = "lu"
            else:
                solver = "full"

        if solver == "eigh":
            components = _pca_eigh(x, num_components)
        elif solver == "qr":
            components = _pca_qr(
                x,
                num_components=num_components,
                num_iterations=self.num_iterations,
                num_oversamples=self.num_oversamples,
                generator=generator,
            )
        elif solver == "lu":
            components = _pca_lu(
                x,
                num_components=num_components,
                num_iterations=self.num_iterations,
                num_oversamples=self.num_oversamples,
                generator=generator,
            )
        else:
            assert solver == "full"
            components = _pca_full(x, num_components)

        index = components.abs().argmax(dim=-2, keepdim=True)
        components *= components.gather(-2, index).sign()

        self._set_buffer("components", components)

    def _transform(self, table: TableTensor) -> TableTensor:
        x = (table.numerical - self.mean) @ self.components
        out = TableTensor(
            columns={Stype.numerical: [f"pca_{i}" for i in range(x.size(-1))]},
            numerical=x,
        )
        return cast(
            TableTensor,
            torch.cat([table.drop_stypes(Stype.numerical), out], dim=-1),
        )

    def __repr__(self, *, indent: int = 0) -> str:
        return (
            f"{' ' * indent}{self.__class__.__name__}("
            f"num_components={self.num_components}, "
            f"solver={self.solver!r}"
            f")"
        )


def _pca_full(x: Tensor, num_components: int) -> Tensor:
    _, _, vh = torch.linalg.svd(x, full_matrices=False)
    return vh[..., :num_components, :].transpose(-2, -1)


def _pca_eigh(x: Tensor, num_components: int) -> Tensor:
    cov = x.transpose(-2, -1) @ x
    _, eigvec = torch.linalg.eigh(cov)
    return eigvec[..., -num_components:].flip(-1)


def _pca_qr(
    x: Tensor,
    num_components: int,
    num_iterations: int,
    num_oversamples: int,
    generator: torch.Generator | None,
) -> Tensor:
    return _pca_randomized(
        x,
        num_components,
        normalize=lambda x: torch.linalg.qr(x, mode="reduced").Q,
        num_iterations=num_iterations,
        num_oversamples=num_oversamples,
        generator=generator,
    )


def _pca_lu(
    x: Tensor,
    num_components: int,
    num_iterations: int,
    num_oversamples: int,
    generator: torch.Generator | None,
) -> Tensor:

    def lu_normalize(x: Tensor) -> Tensor:
        lu, pivot, _ = torch.linalg.lu_factor_ex(x, check_errors=False)
        *_, row, rank = x.size()
        factor = lu.tril(-1)
        factor.diagonal(dim1=-2, dim2=-1).fill_(1)
        swap = torch.arange(row, device=x.device)
        swap = swap.repeat(*x.size()[:-2], rank, 1)
        first = torch.arange(rank, device=x.device)
        first = first.view(*([1] * (x.dim() - 2)), rank, 1)
        first = first.expand(*x.size()[:-2], rank, 1)
        second = (pivot.long() - 1).unsqueeze(-1)
        swap.scatter_(-1, first, second)
        swap.scatter_(-1, second, first)
        while swap.size(-2) > 1:
            count = swap.size(-2)
            even = swap[..., : count - 1 : 2, :]
            odd = swap[..., 1:count:2, :]
            pair = odd.gather(-1, even)
            if count % 2 > 0:
                swap = torch.cat((pair, swap[..., -1:, :]), dim=-2)
            else:
                swap = pair
        order = swap.squeeze(-2).unsqueeze(-1).expand_as(factor)
        return factor.gather(-2, order)

    return _pca_randomized(
        x,
        num_components,
        normalize=lu_normalize,
        num_iterations=num_iterations,
        num_oversamples=num_oversamples,
        generator=generator,
    )


def _pca_randomized(
    x: Tensor,
    num_components: int,
    normalize: Callable[[Tensor], Tensor],
    num_iterations: int,
    num_oversamples: int,
    generator: torch.Generator | None,
) -> Tensor:
    transpose = x.size(-2) < x.size(-1)
    x = x.transpose(-2, -1) if transpose else x

    q = min(num_components + num_oversamples, *x.size()[-2:])
    basis = torch.randn(
        (*x.size()[:-2], x.size(-1), q),
        dtype=x.dtype,
        device=x.device,
        generator=generator,
    )
    for _ in range(num_iterations):
        basis = normalize(x @ basis)
        basis = normalize(x.transpose(-2, -1) @ basis)
    basis, _ = torch.linalg.qr(x @ basis, mode="reduced")
    u, _, vh = torch.linalg.svd(basis.mT @ x, full_matrices=False)

    if transpose:
        return basis @ u[..., :num_components]
    return vh.mT[..., :num_components]
