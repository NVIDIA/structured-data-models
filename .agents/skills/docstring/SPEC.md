# Public Docstring Standard Proposal

Status: Proposed

This document proposes the public docstring standard for `sdm`. It is intentionally separate from `SKILL.md` while the rules are under review. Once accepted, the normative rules should be folded into the skill and backed by repository checks.

## Goals

- Keep public API documentation concise, accurate, and consistent.
- Describe user-visible contracts without exposing incidental implementation details.
- Make objective requirements enforceable before changes reach `main`.
- Give human and agent contributors the same versioned source of guidance.

## Non-goals

- Require a long docstring for every object.
- Document private helpers or internal state.
- Duplicate information already expressed precisely by type annotations.
- Add defensive runtime validation merely to match a docstring.
- Generate semantic descriptions from code or an LLM and treat them as authoritative.

## Normative language

`MUST`, `SHOULD`, and `MAY` describe required, recommended, and optional behavior. Rules have stable identifiers so they can be discussed individually during review.

## Scope and ownership

- **DOC-01 (MUST):** An object is part of the public API when it is exported through the documented `__all__` hierarchy. Public API and reference documentation SHOULD remain aligned.
- **DOC-02 (MUST):** Every public class, function, and non-inherited public method has a docstring.
- **DOC-03 (MUST):** Public constructor parameters are documented in the class docstring, not duplicated in `__init__`.
- **DOC-04 (MUST):** An overriding method does not duplicate an inherited docstring when the inherited contract applies unchanged. Use a targeted `# noqa: D102` in that case. If the override changes public behavior, it MUST document the changed contract.
- **DOC-05 (SHOULD):** Individual implementation modules do not have module docstrings. Package `__init__.py` files MAY have a short package summary.

## Content and structure

- **DOC-06 (MUST):** Begin with a concise, self-contained summary sentence. Function and method summaries use the imperative form, such as `Return the encoded table.` Class summaries state the class purpose directly and avoid phrases such as `This class ...`.
- **DOC-07 (MUST):** Use an `Args:` section when the documented callable or constructor has public parameters. Document every public parameter and no parameters absent from the signature. Include `*args` and `**kwargs` when they are part of the public contract.
- **DOC-08 (MUST):** Use `Returns:` when a return value is not completely described by the summary. Use `Yields:` for yielded values. Omit both for callables that only return `None`.
- **DOC-09 (MUST):** Use `Raises:` only for exceptions that are part of the supported public contract. Do not document incidental downstream failures or exceptions caused solely by violating an already documented precondition.
- **DOC-10 (SHOULD):** Add `Attributes:` for public fitted or learned state that users are expected to inspect. Do not document private caches or incidental state.
- **DOC-11 (SHOULD):** Add `See Also:` when a closely related public API would help the user choose the correct operation. Each entry includes a short explanation of the relationship; do not add link-only lists.
- **DOC-12 (SHOULD):** Add one minimal, deterministic example for a new top-level public processor or another API whose composition or lifecycle is not obvious. Do not add examples to every trivial method.
- **DOC-13 (MUST):** Keep API docstrings focused on the reference contract. Extended tutorials, motivation, comparisons, and end-to-end workflows belong in narrative documentation.

## Parameters, defaults, and types

- **DOC-14 (MUST):** Describe parameter semantics rather than restating type annotations. Mention accepted dtype families, shapes, units, ranges, or supported values only when they affect correct use.
- **DOC-15 (MUST):** Explain what `None`, sentinel values, and special literals mean. Do not merely repeat `default=None`.
- **DOC-16 (MUST):** Document implicit caps, fallbacks, transformations, and side effects that can change results or surprise a caller.
- **DOC-17 (MUST):** Documentation does not promise broader input support than the implementation provides and does not introduce runtime validation requirements that the public API does not need.

## Tensor and processor contracts

- **DOC-18 (MUST):** Tensor parameters and returns include their relevant shape in double-backtick notation, for example `[..., S, C]`. Use a leading `...` for batch dimensions and spell out remaining dimension letters consistently.
- **DOC-19 (MUST):** Document dtype and device behavior when an operation restricts, promotes, converts, or relocates data. Do not enumerate dtypes when the operation transparently preserves or accepts all supported dtypes.
- **DOC-20 (MUST):** Stateful processors document whether `fit()` is required, which public state is learned, and whether a subsequent `fit()` replaces that state, whenever this is not fully defined by the base-class contract.
- **DOC-21 (MUST):** Document user-visible handling of missing or non-finite values, stochastic behavior and generator control, column order or names, stypes, and schema changes when relevant to the processor.
- **DOC-22 (MUST):** Input and output descriptions state whether container structure, dtype, device, shape, or semantic column type changes when that behavior is not obvious from the API.

## Markup, citations, and versions

- **DOC-23 (MUST):** Use `r"""..."""` for docstrings containing math, LaTeX, or backslashes.
- **DOC-24 (MUST):** When an API implements functionality proposed in an academic paper, cite the paper in the first sentence using a stable paper URL where possible.
- **DOC-25 (SHOULD):** Prefer resolvable Sphinx cross-references over plain literals for public internal and intersphinx-mapped external targets.
- **DOC-26 (MUST):** Use `.. deprecated::`, `.. versionchanged::`, or `.. versionadded::` when the corresponding public lifecycle information must remain visible in the API reference.
- **DOC-27 (MUST):** Use a targeted `# noqa: <code>` only when the rule is intentionally inapplicable. Broad or unexplained suppressions are not allowed.

## Example requirements

An example required by DOC-12:

- **EX-01 (MUST):** Runs without network access and on CPU unless the documented feature specifically requires another environment.
- **EX-02 (MUST):** Is deterministic or explicitly controls its randomness.
- **EX-03 (MUST):** Shows public API usage rather than private setup or internal state.
- **EX-04 (MUST):** Is executable by the documentation doctest build unless it is explicitly marked and justified as non-executable.
- **EX-05 (SHOULD):** Demonstrates the smallest realistic input that makes the output contract clear.

## Generation strategy

- **GEN-01 (MUST):** Public contract descriptions are reviewed source text. Generated API pages may consume docstrings, signatures, type annotations, and autosummary metadata, but generated prose is not authoritative without review.
- **GEN-02 (SHOULD):** Reusable parameter descriptions or templates are introduced only for semantics that are genuinely identical across multiple APIs.
- **GEN-03 (MUST):** A generated description has one version-controlled source of truth and a consistency check covering signatures, defaults, and documented parameters.
- **GEN-04 (MUST):** LLM or agent generation may assist authors, but successful generation or skill invocation is not a merge criterion. The resulting documentation must satisfy the same review and CI checks as human-written text.

## Enforcement model

The skill explains how to write docstrings. CI defines the reproducible merge requirements. Skill invocation alone cannot guarantee conformance because it is not reliably observable for every human, editor, or agent workflow.

- **ENF-01 (MUST):** `AGENTS.md` directs contributors and agents changing public API under `sdm/` to read `.agents/skills/docstring/SKILL.md` before acting.
- **ENF-02 (MUST):** Ruff with the repository's Google docstring convention runs over all Python files in required pull-request CI.
- **ENF-03 (MUST):** A Google-style contract checker runs over all of `sdm` and validates documented arguments, constructor parameters, returns, yields, and supported raises against code. `pydoclint` is the proposed general-purpose implementation.
- **ENF-04 (MUST):** A small repository-owned checker validates SDM-specific objective rules that general-purpose linters cannot express, including the public `__all__` surface and class-level constructor documentation.
- **ENF-05 (MUST):** Sphinx HTML builds with warnings treated as errors so malformed markup and unresolved references block a merge.
- **ENF-06 (MUST):** Sphinx doctests run in required pull-request CI so executable examples cannot drift from behavior.
- **ENF-07 (MUST):** The checks in ENF-02 through ENF-06 are required by branch protection before merging to `main`; running them only after a push to `main` is insufficient.
- **ENF-08 (MUST):** Semantic requirements that cannot be checked reliably, such as whether a description captures surprising behavior or cites the correct paper, remain explicit review checklist items.
- **ENF-09 (SHOULD):** Full-repository checks are preferred over diff-only checks when runtime permits, because signature and export changes can invalidate documentation outside the edited lines.

## Adoption

1. Review and agree on this proposal.
2. Fold accepted rules into `SKILL.md` and add the explicit skill-routing rule to `AGENTS.md`.
3. Run the proposed contract checks across `sdm` and classify existing findings.
4. Fix the existing public API where practical. If immediate cleanup is too large, commit a temporary baseline that permits existing violations but rejects new or worsened violations.
5. Add the checks to pre-commit where they are fast enough and to required pull-request CI in all cases.
6. Remove any temporary baseline incrementally.

## Review decisions

The following decisions should be resolved before this proposal becomes normative:

1. Should DOC-12 require an example for every new top-level public processor, or only when a reviewer considers usage non-obvious?
2. Should `__all__` be the sole definition of public API for DOC-01, or should inclusion in the generated API reference also be required?
3. Should ENF-03 adopt `pydoclint`, or should all contract checks live in a repository-owned checker?
4. Which `Raises:` checks should be enabled without encouraging defensive validation or making unsupported inputs part of the public contract?
5. Can the current tree satisfy the new checks immediately, or is a temporary baseline required?
6. Should DOC-19 require explicit dtype and device statements for every tensor operation, or only for restrictions and transformations as proposed?

## References and selection rationale

- [PEP 257](https://peps.python.org/pep-0257/) defines the Python-level structure and summary conventions on which other standards build.
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html#s3.8.1-comments-in-doc-strings) matches the existing SDM `Args:`, `Returns:`, `Yields:`, and `Raises:` syntax.
- [numpydoc validation](https://numpydoc.readthedocs.io/en/stable/validation.html) demonstrates versioned, configurable docstring validation in pre-commit and Sphinx.
- [pandas documentation guidelines](https://pandas.pydata.org/docs/dev/development/contributing_documentation.html) demonstrate strict Sphinx validation and executable examples for a large public API.
- [scikit-learn docstring guidelines](https://scikit-learn.org/stable/developers/contributing.html#guidelines-for-writing-docstrings) are relevant to SDM's estimator- and processor-like APIs, especially shapes, dtypes, defaults, fitted attributes, and examples.
- [JAX doctest guidelines](https://github.com/jax-ml/jax/blob/main/docs/developer.md#doctests) demonstrate executable documentation for a modern tensor and accelerator library.
- [PyTorch's docstring linter](https://github.com/pytorch/pytorch/blob/main/tools/linter/adapters/docstring_linter.py) demonstrates gradual enforcement through a grandfathered baseline in SDM's closest runtime ecosystem.
- [Transformers auto-docstrings](https://github.com/huggingface/transformers/blob/main/docs/source/en/auto_docstring.md) demonstrate signature- and template-based generation for a large model and processor zoo, while also showing why generation is most useful for repeated semantics.
- [`pydoclint`](https://jsh9.github.io/pydoclint/) supports Google-style signature, argument, return, yield, raise, constructor, and attribute consistency checks and can be adopted incrementally with a baseline.
