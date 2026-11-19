---
name: draft-release-note
description: Draft and revise SDM minor-release notes for vX.Y.0 from repository changes, with concise highlights, breaking changes, and runnable examples. Use for minor releases only, not patch or major releases.
---

# Draft Release Note

Prepare researcher-facing SDM release notes for a stable minor release such as `v0.3.0`. Accept only `vX.Y.0` targets with a nonzero minor version; clarify the intended scope for patch, major, or prerelease targets rather than silently applying this workflow.

## Establish the release scope

- Read repository instructions and inspect the current branch, tags, existing release drafts, and any gist already identified in the conversation. Preserve ongoing edits; do not regenerate an existing draft from scratch.
- Establish the target tag or explicitly chosen commit and the preceding stable release, including its latest patch (for example, `v0.2.1...v0.3.0`). Exclude prerelease tags when choosing the stable baseline. If the target is not tagged yet, use the agreed branch or commit and distinguish planned links from verified ones.
- Inspect the comparison, merged PRs, public API changes, examples, documentation, and benchmarks. Use connected GitHub tools when available and sufficient; otherwise use `gh` and local Git. Ground claims in the release range, not newer documentation or unrelated products.
- Verify package ownership and availability at the target revision. In particular, do not describe a model from a separate package as an SDM feature unless this release actually integrates it. Distinguish time-series anomaly detection from forecasting.

## Shape the draft

Use the maintainer's chosen draft destination and existing naming conventions; do not prescribe a local filename. Start with a short release announcement followed by a linked contents list and these sections:

- **Highlights:** Select the substantive capabilities researchers can now use. Explain the capability and its purpose before API details; use a short example where it makes adoption easier.
- **Breaking Changes:** Describe affected public APIs and the concrete migration, with relevant PR links. Omit this section and its contents entry when there are no breaking changes.
- **Full Changelog:** Link the exact baseline-to-target GitHub comparison.

Do not add a generic Features section, a processor inventory, or a documentation-expansion list by default. Include those only when the user requests them or they are a substantive highlight. Remove corresponding contents links when removing sections.

Write concise, concrete prose. Prefer “For more details:” when introducing supporting references. Link meaningful API names, guides, training code, model cards, papers, and benchmarks where they support the claim. Prefer release-pinned repository links; verify documentation paths and do not invent paper identifiers, performance results, or URLs.

## Examples and output

- Use the simplest public API available in the target release. Include the imports and setup needed to run the example, with descriptive variables such as `features` and `target`.
- Verify examples in a suitable environment when practical. Report material verification limits instead of implying execution.
- Show printed output as Python comments immediately after each corresponding `print(...)` call, inside the same code fence. Prefer actual output; representative output is acceptable when clearly identified as illustrative or variable. Preserve the real display format rather than paraphrasing it as a shape description.
- For `TableTensor`, show its size and semantic column blocks, using the representation at the target revision. Randomly sampled feature assignments can vary; do not add a seed solely to force the illustration to match.

Example layout:

```python
print(target)
# TableTensor(
#   size=(1024, 1),
#   blocks={
#     numerical (1): [target],
#   },
# )
```

## Draft destination and synchronization

Use an existing draft or the destination requested by the maintainer. A gist is optional. Once gist synchronization is authorized, keep it synchronized after each edit, including small wording changes; do not stop after editing only a local working copy.

- Reuse the gist identified in the session. When asked to update an existing gist without a link, inspect available gists for the matching release draft; ask only if the destination remains ambiguous. Never hard-code a gist ID from an earlier release.
- Inspect the gist filename and content before writing. If it has diverged, reconcile the changes rather than overwriting remote edits. Update only the intended release-note file and preserve the gist's visibility, description, and other files.
- Create a new draft gist only when requested or otherwise authorized. Gist sync does not authorize publishing a GitHub release, changing tags, or posting to issues or PRs; follow repository review requirements for those actions.
- When using `gh`, pass the actual gist ID, remote filename, and working-copy path to `gh gist edit`. Fetch the named file through the gist API afterward and compare its content with the intended draft. Use structured payloads or files for multiline text, not interpolated shell text.
- If a write fails, report that the gist remains unsynchronized. After an uncertain result, read it back before retrying. Do not claim success until the remote content is verified.

## Finish the iteration

Check the final draft against the release evidence, public API signatures, Markdown rendering, contents links, and the user's latest edits. Apply requested removals consistently to the local draft and gist; do not restore removed material on later revisions. If a follow-up is ambiguous, do not silently reverse an explicit removal.

Report the draft location, any authorized gist link and synchronization status, and material unverified claims. Keep the response brief. Commit or push repository changes only within the user's authorization.

## Final checklist

- [ ] The target is a stable minor release (`vX.Y.0`), and the comparison starts at the preceding stable release, including its latest patch.
- [ ] Highlights and breaking changes are grounded in the release range and describe public capabilities or concrete migrations.
- [ ] Model tasks, package ownership, API signatures, and supporting links are accurate for the target revision.
- [ ] Examples use public APIs, output comments follow each print call, and illustrative output or execution limits are identified.
- [ ] Removed sections stay removed, and contents links match the final headings.
- [ ] The draft uses the maintainer's chosen destination; no personal paths, filenames, or gist IDs are assumed.
- [ ] If gist synchronization is authorized, the intended file is updated and its remote content is verified; otherwise no gist is created or modified.
- [ ] Remaining verification limits and the draft location are reported; publication, commits, and pushes stay within authorization.
