from typing import Any, cast

import torch
from torch import Tensor
from torch.nn import GELU, Linear, Sequential


class QASSMax(torch.nn.Module):
    r"""Learnable query scaler used by query-aware scalable softmax (QASSMax)
    introduced in the `"TabICLv2: A better, faster, scalable, and
    open tabular foundation model" <https://arxiv.org/abs/2602.11139>`_ paper
    as a temperature-scaling method for attention.

    For a query tensor ``q`` and key length ``n``, this module returns a scaled
    query

    .. math::

        \tilde{q}_{hi} = q_{hi} \cdot
          \mathrm{MLP}_{\mathrm{base}}(\log n)_{hi} \cdot
          \left(1 + \tanh(\mathrm{MLP}_{\mathrm{gate}}(q_h)_i)\right),

    where ``h`` indexes attention heads, ``i`` indexes head channels,
    ``MLP_base`` maps the log key length to per-head, per-channel scale
    factors, and ``MLP_gate`` maps each per-head query vector to a bounded
    query-dependent gate. In this implementation, ``n`` is clamped to at least
    1 before taking ``log``. The gate is initialized as identity modulation by
    zero-initializing the final layer of ``MLP_gate``.

    Multiplying the query scales the subsequent attention logits while keeping
    the attention computation compatible with standard softmax kernels. The
    length-dependent factor counteracts attention fading as the number of keys
    grows, while the query-dependent gate lets the scale vary across queries.

    Args:
        channels: The number of channels per attention head.
        num_heads: The number of attention heads.
        hidden_channels: The hidden width of the scale and gate MLPs.
        device: The device to use for module parameters.
        dtype: The dtype to use for module parameters.
    """

    def __init__(
        self,
        channels: int,
        num_heads: int,
        hidden_channels: int = 64,
        device: torch.device | None = None,
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
        key_len: Tensor | int,  # [..., 1] or [..., S] or scalar
    ) -> Tensor:  # [..., S, H, C]
        r"""Forward pass of :class:`QASSMax`.

        Args:
            query: The query tensor to scale, with shape ``[..., S, H, C]``.
                ``S`` is the query sequence length, ``H`` is the number of
                attention heads, and ``C`` is the channels per head.
            key_len: The number of valid keys used to scale each query.

                Supported forms are:

                * ``int``: one shared key length for every query.
                * tensor shaped ``[..., 1]``: one key length per batch/context
                  item, shared across all query positions in ``S``.
                * tensor shaped ``[..., S]``: one key length per query
                  position, such as the result of reducing an attention mask
                  over its key dimension.

        Returns:
            The scaled query tensor.
        """
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
