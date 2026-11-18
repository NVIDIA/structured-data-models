# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from torch import Tensor
from torch.nested._internal.nested_tensor import NestedTensor

from sdm.prior import HP, Correlated, LogUniformInt, Mixture
from sdm.prior._cdist import cdist_argmin
from sdm.prior.converters import CategoricalConverter
from sdm.prior.functions import RandomFunction


class NearestNeighborConverter(CategoricalConverter):
    """Assign each row to its nearest sampled center using Minkowski distance.

    Args:
        p: Minkowski distance exponent.
        output_mode: Representation passed to child nodes. ``"identity"``
            keeps input slices, ``"index"`` repeats category indices across
            channels,
            ``"center"`` returns the nearest center, and ``"function"``
            applies ``random_fn`` to that center.
        random_fn: Random function applied to selected centers. Required
            for ``"function"`` output mode.
    """

    def __init__(
        self,
        p: HP[float] | Correlated[float] | float,
        output_mode: str | HP[str] = "identity",
        *,
        random_fn: RandomFunction | None = None,
    ) -> None:
        self.p = HP.from_value(p)
        self.output_mode = HP.from_value(output_mode)
        self._random_fn = random_fn
        self.use_full_slice = Mixture(
            [False, True],
            correlated=True,
            key="CategoricalConverter.use_full_slice",
        )

    def forward(  # noqa: D102
        self,
        slices: Tensor,  # [num_cols, num_categories, num_rows]
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, Tensor]:
        x = slices
        x_in = x

        offsets: Tensor | None = None
        count: Tensor | None = None
        if isinstance(x, NestedTensor):
            offsets = x.offsets()
            count = offsets.diff()
            x = x.to_padded_tensor(0.0)[:, : x._max_seqlen]

        num_cols, max_k, num_rows = x.size()

        x = x.transpose(1, 2).contiguous()  # [C, R, max_k]
        cardinalities = (
            count.tolist() if count is not None else [max_k] * num_cols
        )
        ps = self.p.sample_batch(size=num_cols, generator=generator).tolist()

        outs = []
        center_indices = []
        for c, (k, p_c) in enumerate(zip(cardinalities, ps)):
            x_c = x[c, ..., :k]
            if self.use_full_slice.pick(generator=generator) or k == 1:
                x_distance = x_c
            else:
                d = LogUniformInt(1, k - 1).sample(generator=generator)
                dims = torch.randperm(k, device=x.device, generator=generator)[
                    :d
                ]
                x_distance = x_c[..., dims]

            # Sample distinct center rows.
            center_index = torch.rand(
                num_rows, device=x.device, generator=generator
            ).argsort()[:k]
            centers = x_distance[center_index]
            outs.append(cdist_argmin(x_distance, centers, float(p_c)))
            center_indices.append(center_index)

        out = torch.stack(outs)  # [C, R]

        output_mode = self.output_mode.sample(generator=generator)
        if output_mode == "identity":
            return out, x_in

        if output_mode == "index":
            mod = out.to(x.dtype).unsqueeze(1).expand(-1, max_k, -1)
        else:
            assert output_mode in ("center", "function")

            mods = []
            for c, k in enumerate(cardinalities):
                index = center_indices[c][out[c]]
                mod_c = x[c, :, :k][index].T.contiguous()
                if output_mode == "function":
                    assert self._random_fn is not None, (
                        "output_mode='function' requires random_fn"
                    )
                    # Keep dense views column-major for consistent
                    # floating-point accumulation.
                    values = mod_c.T
                    if offsets is not None:
                        values = values.contiguous()
                    mod_c = self._random_fn.forward(
                        x=values,
                        out_channels=k,
                        generator=generator,
                    ).T.contiguous()
                mods.append(mod_c)

            if offsets is not None:
                mod = torch.nested.nested_tensor_from_jagged(
                    values=torch.cat(mods), offsets=offsets
                )
            else:
                mod = torch.stack(mods)

        if (
            offsets is not None
            and count is not None
            and not isinstance(mod, NestedTensor)
        ):
            mask = torch.arange(max_k, device=x.device)
            mask = mask.view(1, -1) < count.view(-1, 1)  # [C, max_k]
            mod = torch.nested.nested_tensor_from_jagged(
                values=mod[mask],
                offsets=offsets,
            )

        return out, mod
