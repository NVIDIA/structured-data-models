---
name: docstring
description: Write or review docstrings for public modules, classes, and functions in sdm/. Use when adding or reviewing docstrings to follow repo best practices (Args sections, paper citations, tensor shape notation).
---

# Docstring Writing and Reviewing Guide

Use this when writing or reviewing docstrings for public modules, classes, and functions in `sdm/`.
Follow these best practices up front to keep docstrings consistent across the codebase.

## General Principles

- Keep docstrings as minimal as possible and avoid documenting implementation details.
- Do not add module-level docstrings to individual modules; keep only a short package summary in the package `__init__.py`.
- Every public class and function has a docstring, a one-line summary, then an `Args:` section.
- When a class or function implements functionality proposed in an academic paper, cite it in the first sentence of its docstring.
- Document every public parameter, especially, constructor parameters. Document them in the **class** docstring's `Args:`, not in `__init__`.
- Use `r"""..."""` whenever the docstring contains math, LaTeX, or backslashes (e.g., a `.. math::` block).
- Describe tensor parameters with their shape in double-backtick notation, using a leading `...` for the batch dimensions (e.g., `[..., S, H, C]`). Spell out each remaining dimension letter, and keep the notation consistent across related processors/modules.
- If splitting a long line leads to a line-too-long error, put `# noqa: <code>` to ignore the error.
- Document non-obvious behavior: implicit caps, defaults, transformations, or side effects that affect results. If it would surprise a caller, state it.
- Do not add docstrings to methods that already have one in superclass's methods even if the class/methods are public. For example, `Processor.transform()` already has a general docstring that's applicable to its subclasses. In this case, put a comment `# noqa: D102` to ignore the linter error.

## Example

```python
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

The same roles resolve internal `sdm` targets and external ones. Prefer
cross-reference roles over plain literals, e.g.,

```python
""":class:`pandas.DataFrame`"""
```

over

```python
"""``pandas.DataFrame``"""
```

whenever the target lives in an intersphinx-mapped project (`python`, `torch`, `numpy`,
`pandas`, `pyarrow`, `cudf`, `typing_extensions`; see `intersphinx_mapping`
in `docs/source/conf.py`).

```python
r"""
- :mod:`sdm.nn`: Module reference.
- :class:`~sdm.nn.Module`: Class reference. External targets use the same
  role: :class:`pandas.DataFrame`, :class:`torch.nn.Module`, :class:`dict`,
  :class:`TypeError`, :class:`collections.abc.Mapping`.
- :class:`~pyarrow.Array`: Leading ``~`` renders only ``Array``. Use it in
  summary lines; keep the full path in ``Args:`` entries.
- :meth:`~sdm.nn.Module.forward`: Method reference.
- :func:`torch.nn.functional.scaled_dot_product_attention`: Function
  reference. Use :func: for free functions and :meth: only for methods.
- :attr:`attribute`: Attribute reference.
- :math:`equation`: Inline math.
- :ref:`label`: Internal label reference.
- :ref:`calling convention <torch-dispatch-calling-convention>`: External
  label reference with custom link text, resolved through intersphinx.
- :external+torch:ref:`torch.int32 <dtype-doc>`: Label reference pinned to a
  specific project's inventory. Use for targets without their own API entry
  (e.g., dtypes like ``torch.int32`` resolve to the ``dtype-doc`` label).
  Never guess label names: search the linked documentation for a fitting
  target by dumping the project's inventory, e.g.
  ``uv run --extra doc python -m sphinx.ext.intersphinx https://docs.pytorch.org/docs/stable/objects.inv | grep -i dtype``.
"""
```

## Verification

Run `uv run ruff check`. This is a structural backstop only: it flags missing public class/method/function docstrings, capitalization, end punctuation, `r"""` for backslashes, and `Args:`/signature mismatches.
It does **not** require an `Args:` section to exist, or verify citations, shape notation, or behavior notes — confirm those by hand against the principles above.
