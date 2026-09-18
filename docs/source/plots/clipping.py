# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Clipping figures rendered by the Sphinx plot directive."""

import matplotlib
import torch

import sdm
import sdm.processing as sp

matplotlib.use("Agg")

import matplotlib.pyplot as plt


@torch.inference_mode()
def plot(name: str) -> None:
    """Plot public processor outputs on a fixed input grid."""
    x = torch.linspace(-8, 8, 1601, dtype=torch.float64)
    query = sdm.TableTensor.from_tensor(x[:, None])
    # Fit only on this separate, deterministic single-column context.
    context = sdm.TableTensor.from_tensor(
        torch.linspace(-2, 2, 401, dtype=torch.float64)[:, None],
    )
    figures = [
        (
            "clip",
            "Clip",
            [(f"bounds = ±{b:g}", sp.Clip(-b, b)) for b in (0.5, 1, 2)],
            (-2.5, 2.5),
        ),
        (
            "clip_sigma",
            "ClipSigma",
            [
                (f"threshold = {t:g}", sp.ClipSigma(threshold=t))
                for t in (1, 2, 3)
            ],
            (-6, 6),
        ),
        (
            "clip_quantiles",
            "ClipQuantiles",
            [
                (
                    f"quantiles = ({q:g}, {1 - q:g})",
                    sp.ClipQuantiles(q_low=q, q_high=1 - q),
                )
                for q in (0.375, 0.25, 0.0)
            ],
            (-2.5, 2.5),
        ),
    ]
    with plt.rc_context({"font.size": 11, "svg.hashsalt": "sdm-clipping"}):
        for filename, title, processors, ylim in figures:
            if filename != name:
                continue
            fig, ax = plt.subplots(figsize=(9, 4.6), layout="constrained")
            fig.set_facecolor("white")
            ax.set_facecolor("white")
            ax.set_axisbelow(True)
            ax.grid(color="#e8edf3", linewidth=0.8)
            ax.axhline(0, color="#cbd5e1", linewidth=1)
            ax.axvline(0, color="#cbd5e1", linewidth=1)
            for (label, processor), color in zip(
                processors, ("#099981", "#2878ce", "#974cbb")
            ):
                processor.fit(context)
                y = processor.transform(query).numerical[:, 0]
                ax.plot(x, y, color=color, linewidth=2.6, label=label)
                if isinstance(processor, sp.Clip):
                    bounds = (processor.min_value, processor.max_value)
                else:
                    bounds = (
                        processor.lower_bound.item(),
                        processor.upper_bound.item(),
                    )
                for bound in bounds:
                    ax.axhline(
                        bound, color=color, linestyle=(0, (2, 4)), alpha=0.35
                    )
            ax.plot(
                x, x, color="#94a3b8", linestyle=(0, (5, 4)), label="y = x"
            )
            ax.set(
                title=title,
                xlabel="Input x",
                ylabel="Transformed value",
                xlim=(-8, 8),
                ylim=ylim,
            )
            ax.spines[["top", "right"]].set_visible(False)
            ax.spines[["bottom", "left"]].set_color("#cbd5e1")
            ax.legend(
                loc="lower center",
                bbox_to_anchor=(0.5, 1.12),
                ncol=2,
                frameon=False,
            )


def clip() -> None:
    """Draw fixed-interval clipping."""
    plot("clip")


def clip_sigma() -> None:
    """Draw fitted sigma clipping."""
    plot("clip_sigma")


def clip_quantiles() -> None:
    """Draw fitted quantile clipping."""
    plot("clip_quantiles")
