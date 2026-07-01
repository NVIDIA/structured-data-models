---
name: docstring
description: Write or review docstrings for public modules, classes, and functions in sdm/. Use when adding or reviewing docstrings to follow repo best practices (Args sections, paper citations, tensor shape notation).
---

# Docstring Writing and Reviewing Guide

Use this when writing or reviewing docstrings for public modules, classes, and functions in `sdm/`.
Follow these best practices up front to keep docstrings consistent across the codebase.

## General Principles

- Every module starts with a one-line module docstring, a single declarative line summarizing the module's purpose.
- Every public class and function has a docstring, a one-line summary, then an `Args:` section.
- When a class or function implements functionality proposed in an academic paper, cite it in the first sentence of its docstring.
- Document every public parameter, especially, constructor parameters. Document them in the **class** docstring's `Args:`, not in `__init__`.
- Use `r"""..."""` whenever the docstring contains math, LaTeX, or backslashes (e.g., a `.. math::` block).
- Describe tensor parameters with their shape in double-backtick notation, using a leading `...` for the batch dimensions (e.g., `[..., S, H, C]`). Spell out each remaining dimension letter, and keep the notation consistent across related processors/modules.
- If splitting a long line leads to a line-too-long error, put `# noqa: <code>` to ignore the error.
- Document non-obvious behavior: implicit caps, defaults, transformations, or side effects that affect results. If it would surprise a caller, state it.

## Example

```python
"""Scaled normalization transforms for structured-data models."""


class MyClass:
    r"""My Class from the `"My Paper" <https://arxiv.org/abs/2602.11139>`_ paper.

    .. math::

        y = \frac{x}{\sqrt{d}}

    Args:
        fill_value: Value written into masked positions. Capped at ``1.0``;
            larger values are silently clamped.
    """

    def __init__(self, fill_value: float) -> None:
        ...

    def forward(self, x: Tensor) -> Tensor:
        """Normalize ``x`` along its last dimension.

        Args:
            x: Input tensor with shape ``[..., S, C]``.

        Returns:
            Tensor with shape ``[..., S, C]``.
        """

    def citation(self) -> str:
        """Return the key from :func:`sdm.refs.resolve_default_citation_key`."""  # noqa: E501
```

## Sphinx Cross-Referencing Reference

```python
r"""
- :mod:`sdm.nn`: Module reference.
- :func:`~sdm.nn.Module.forward`: Function reference.
- :class:`~sdm.nn.Module`: Class reference.
- :meth:`~sdm.nn.Module.forward`: Method reference.
- :attr:`attribute`: Attribute reference.
- :math:`equation`: Inline math.
- :ref:`label`: Internal reference.
"""
```

## Verification

Run `uv run ruff check`. This is a structural backstop only: it flags missing public class/method/function docstrings, capitalization, end punctuation, `r"""` for backslashes, and `Args:`/signature mismatches.
It does **not** require an `Args:` section to exist, or verify citations, shape notation, or behavior notes — confirm those by hand against the principles above.
