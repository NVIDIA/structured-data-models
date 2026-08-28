import abc
from typing import Any, cast

import torch
from torch import Tensor
from torch.nn import GELU, Linear, Parameter, Sequential


class QueryScaling(torch.nn.Module, abc.ABC):
    r"""Base class for query scaling modules in :class:`SDPA`.

    Query scaling modules transform projected query heads before scaled
    dot-product attention. They may use the effective key length to implement
    length-aware temperature scaling while preserving compatibility with
    :func:`torch.nn.functional.scaled_dot_product_attention`.
    """

    def __init__(self) -> None:
        super().__init__()

    @abc.abstractmethod
    def forward(
        self,
        query: Tensor,  # [..., S, H, C]
        *,
        key_len: Tensor | int,  # [..., 1] or [..., S] or scalar
    ) -> Tensor:  # [..., S, H, C]
        r"""The forward pass.

        Args:
            query: The query tensor to scale, with shape ``[..., S, H, C]``.
                ``S`` is the query sequence length, ``H`` is the number of
                attention heads, and ``C`` is the channels per head.
            key_len: The number of valid keys used to scale each query.
                An ``int`` denotes one shared key length for every query.
                A tensor shaped ``[..., 1]`` denotes one key length per batch
                item, shared across all query positions in ``S``.
                A tensor shaped ``[..., S]`` denotes one individual key length
                per query position.

        Returns:
            Tensor with shape ``[..., S, H, C]``.
        """


class QASSMax(QueryScaling):
    r"""Query-Aware Scalable SoftMax (QASSMax).

    This scaling method was introduced in the `"TabICLv2: A better, faster,
    scalable, and open tabular foundation model"
    <https://arxiv.org/abs/2602.11139>`_ paper as a temperature-scaling method
    for attention.

    For a query tensor :math:`q` and key length :math:`n`, this module returns
    a scaled query

    .. math::

        \tilde{q}_{hi} = q_{hi} \cdot
          \mathrm{MLP}_{\mathrm{base}}(\log n)_{hi} \cdot
          \left(1 + \tanh(\mathrm{MLP}_{\mathrm{gate}}(q_h)_i)\right),

    where :math:`h` indexes attention heads, :math:`i` indexes head channels,
    :math:`\mathrm{MLP}_{\mathrm{base}}` maps the log key length to per-head,
    per-channel scale factors, and :math:`\mathrm{MLP}_{\mathrm{gate}}` maps
    each per-head query vector to a bounded query-dependent gate.

    Multiplying the query scales the subsequent attention logits while keeping
    the attention computation compatible with standard softmax kernels.
    The length-dependent factor counteracts attention fading as the number of
    keys grows, while the query-dependent gate lets the scale vary across
    queries.

    Args:
        channels: The number of channels per attention head.
        num_heads: The number of query attention heads.
        hidden_channels: The hidden width of the scale and gate MLPs.
        device: The device.
        dtype: The dtype.
    """

    def __init__(
        self,
        channels: int,
        num_heads: int,
        hidden_channels: int = 64,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}

        self.scale = Sequential(
            Linear(1, hidden_channels, **factory_kwargs),
            GELU(),
            Linear(hidden_channels, num_heads * channels, **factory_kwargs),
        )
        self.gate = Sequential(
            Linear(channels, hidden_channels, **factory_kwargs),
            GELU(),
            Linear(hidden_channels, channels, **factory_kwargs),
        )

        torch.nn.init.zeros_(cast(Linear, self.gate[-1]).weight)
        torch.nn.init.zeros_(cast(Linear, self.gate[-1]).bias)

    def forward(
        self,
        query: Tensor,  # [..., S, H, C]
        *,
        key_len: Tensor | int,  # [..., 1] or [..., S] or scalar
    ) -> Tensor:  # [..., S, H, C]
        r""":meta private:"""  # noqa: D415
        if isinstance(key_len, Tensor):
            log_key_len = key_len.float().clamp(min=1.0).log().to(query.dtype)
        else:
            one = query.new_ones((), dtype=torch.float32)
            log_key_len = (one * key_len).clamp(min=1.0).log().to(query.dtype)
            log_key_len = log_key_len.view([1] * (query.dim() - 2))
        scale = self.scale(log_key_len.unsqueeze(-1))  # [..., S or 1, H * C]
        scale = scale.unflatten(-1, (query.size(-2), query.size(-1)))
        gate = 1 + self.gate(query).tanh()
        return query * scale * gate


class PerHeadLogNScale(QueryScaling):
    r"""Trainable per-head LogN query scaling.

    Each attention head receives an independent, unconstrained coefficient
    initialized to ``0.43`` and multiplying the logarithm of the effective key
    length. Key lengths are clamped to at least one, and the logarithm is
    computed in fp32.

    Args:
        num_heads: The number of query attention heads.
        device: The device.
        dtype: The dtype.
    """

    def __init__(
        self,
        num_heads: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.head_scale = Parameter(
            torch.full((num_heads,), 0.43, device=device, dtype=dtype)
        )

    def forward(
        self,
        query: Tensor,  # [..., S, H, C]
        *,
        key_len: Tensor | int,  # [..., 1] or [..., S] or scalar
    ) -> Tensor:  # [..., S, H, C]
        r""":meta private:"""  # noqa: D415
        if isinstance(key_len, Tensor):
            log_key_len = key_len.float().clamp(min=1.0).log()
        else:
            log_key_len = (
                query.new_full((), key_len, dtype=torch.float32)
                .clamp(min=1.0)
                .log()
            )
        log_key_len = log_key_len.to(query.dtype)
        head_scale = self.head_scale.to(query.dtype).reshape(
            *((1,) * (query.dim() - 2)), -1, 1
        )
        return query * log_key_len[..., None, None] * head_scale
