# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
from collections.abc import Hashable
from contextvars import ContextVar, Token
from typing import Self, TypeVar

import torch
from torch import Tensor

from sdm.prior.hp import HP

T = TypeVar("T")


def _sample_alpha_beta(
    size: tuple[int, ...],
    generator: torch.Generator | None,
) -> tuple[Tensor, Tensor]:
    eps = 1e-6
    t = torch.rand(size, device="cpu", generator=generator)
    t = t.clamp(eps, 1 - eps)

    log_low = math.log(0.1)
    log_high = math.log(10_000)
    s = torch.rand(size, device="cpu", generator=generator)
    s = (log_low + (log_high - log_low) * s).exp()
    return s * t, s * (1 - t)


class CorrelatedContext:
    """Share sampling parameters within a scope.

    Numerical samplers reuse Beta parameters. Correlated mixtures reuse
    selection probabilities. Values and selected items can still differ
    between draws. Keys share parameters only within the same kind of
    sampling operation.

    Args:
        generator: Pseudorandom generator. Calls within the scope may omit
            it or pass the same object.
    """

    def __init__(
        self,
        *,
        generator: torch.Generator | None = None,
    ) -> None:
        self.generator = generator
        self._cache: dict[Hashable, tuple[Tensor, Tensor]] = {}
        self._mixture_cache: dict[Hashable, Tensor] = {}
        self._token: Token[CorrelatedContext | None] | None = None

    def _resolve_generator(
        self,
        generator: torch.Generator | None,
    ) -> torch.Generator | None:
        if generator is not None and generator is not self.generator:
            raise ValueError(
                "The supplied generator differs from the active "
                "CorrelatedContext's generator. Omit generator to use the "
                "context's generator, or pass the same generator object."
            )
        return self.generator

    def params(
        self, key: Hashable, size: tuple[int, ...]
    ) -> tuple[Tensor, Tensor]:
        """Return the shared Beta parameters for a numerical sampler.

        Args:
            key: Identity of the shared parameter set.
            size: Leading sample shape, excluding the final draw dimension.
        """
        if key not in self._cache:
            self._cache[key] = _sample_alpha_beta(
                size=size,
                generator=self.generator,
            )
        alpha, beta = self._cache[key]
        if alpha.shape != size:
            raise ValueError(
                f"Shared key {key!r} requires leading shape "
                f"{tuple(alpha.shape)}, got {size}."
            )
        return alpha, beta

    def mixture_weight(self, key: Hashable, weights: Tensor) -> Tensor:
        r"""Return shared probabilities drawn from the mixture's weights.

        Samples ``w ~ Dirichlet(s * weight / mean(weight))`` with
        :math:`s \sim \text{LogUniform}[0.1, 3]` sampled once per key and
        shared across components.

        Args:
            key: Identity of the shared mixture probabilities.
            weights: Tensor of relative weights with shape ``[K]`` for
                ``K`` options.
                Shared keys must keep their option order and weights.
        """
        if key not in self._mixture_cache:
            log_low = math.log(0.1)
            log_high = math.log(3.0)
            s = torch.rand((), device="cpu", generator=self.generator)
            s = (log_low + (log_high - log_low) * s).exp()
            alpha = s * weights / weights.mean()
            g = torch._standard_gamma(alpha, generator=self.generator)
            self._mixture_cache[key] = g / g.sum()
        probabilities = self._mixture_cache[key]
        if probabilities.numel() != weights.numel():
            raise ValueError(
                f"Shared key {key!r} requires {probabilities.numel()} "
                f"options, got {weights.numel()}."
            )
        return probabilities

    def __enter__(self) -> Self:
        if self._token is not None:
            raise RuntimeError("This correlation context is already active.")
        self._token = _CONTEXT.set(self)
        return self

    def __exit__(self, *_: object) -> None:
        assert self._token is not None
        _CONTEXT.reset(self._token)
        self._token = None
        self._cache.clear()
        self._mixture_cache.clear()


_CONTEXT: ContextVar[CorrelatedContext | None] = ContextVar(
    "correlated_context", default=None
)


class Correlated(HP[T]):
    """Sample numerical values from a shared, randomly sampled distribution.

    A :class:`~sdm.prior.CorrelatedContext` reuses Beta parameters for each
    key and supplies the generator for sampling.
    Without a context, each call samples new parameters. The Beta draws
    are mapped through ``hp.transform``.

    Args:
        hp: Numerical parameter sampler with a ``transform``
            method mapping quantile levels to values.
        key: Share Beta parameters with other samplers using this key.
            Defaults to this sampler's identity.
    """

    def __init__(
        self,
        hp: HP[T],
        *,
        key: Hashable | None = None,
    ) -> None:
        self.hp = hp
        self.key = key
        if not hasattr(hp, "transform"):
            raise TypeError(
                f"Correlated requires a numerical HP with transform, "
                f"got {type(hp).__name__}."
            )

    def sample_batch(
        self,
        size: int | torch.Size | tuple[int, ...] = (),
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Sample a tensor, sharing parameters along its final dimension.

        Args:
            size: Output shape ``[..., N]`` or vector length. Leading
                dimensions identify independent parameter sets; ``N``
                samples share each set. A scalar uses one parameter set.
                Shared keys must retain their leading shape in a scope.
            generator: Pseudorandom generator. In a context, omit it or pass
                the context's generator.
        """
        size = (size,) if isinstance(size, int) else tuple(size)
        shape = size or (1,)
        if (ctx := _CONTEXT.get()) is not None:
            generator = ctx._resolve_generator(generator)
            alpha, beta = ctx.params(
                self.key if self.key is not None else id(self), shape[:-1]
            )
        else:
            alpha, beta = _sample_alpha_beta(
                size=shape[:-1],
                generator=generator,
            )
        alpha = alpha.unsqueeze(-1).expand(shape)
        beta = beta.unsqueeze(-1).expand(shape)
        gx = torch._standard_gamma(alpha, generator=generator)
        gy = torch._standard_gamma(beta, generator=generator)
        return self.hp.transform((gx / (gx + gy)).reshape(size))  # type: ignore
