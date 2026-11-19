---
name: draft-release-note
description: Draft and revise SDM minor-release notes for vX.Y.0 from repository changes, with highlights, breaking changes, contributor credits, and examples. Use for minor releases only, not patch or major releases.
---

# Draft Release Note

Draft researcher-facing SDM release notes for stable minor releases (`vX.Y.0`, with a nonzero minor version). Clarify scope for patch, major, or prerelease requests.

## Establish the scope

Read repository instructions and preserve existing draft edits. Use the maintainer's chosen destination and naming conventions; do not assume a local filename or gist.

Identify the target tag or agreed commit and the preceding stable release, including its latest patch (for example, `v0.2.1...v0.3.0`). Inspect the comparison, merged PRs, public APIs, examples, and documentation. Ground claims in that range. Verify package ownership and model tasks; do not conflate separately packaged models with SDM integrations or anomaly detection with forecasting.

## Required structure

Use this order and heading hierarchy. Replace placeholders; the number of highlight subsections depends on the release. Include Breaking changes and its contents link only when applicable.

```markdown
We're excited to release SDM vX.Y.0! :tada:

- [Highlights](#highlights)
- [Breaking changes](#breaking-changes)
- [New contributors](#new-contributors)
- [Full changelog](#full-changelog)

# Highlights

## <Descriptive highlight title; repeat for each highlight>

# Breaking changes

# New contributors

# Full changelog
```

- **Highlights (required):** Explain substantive new capabilities and their purpose, each under a descriptive second-level heading. Add a short public-API example where useful. Do not add a generic Features section, processor inventory, or documentation-expansion list unless requested or substantive enough to be a highlight.
- **Breaking changes (if any):** Identify the affected public API, concrete migration, and relevant PR.
- **New contributors (required):** List first-time contributors by GitHub handle with links to their first merged PRs. Verify first-time status against earlier history or GitHub metadata. If none, say so; if verification is blocked, retain the section and report the gap rather than inventing credits.
- **Full changelog (required):** Link the exact baseline-to-target GitHub comparison.

## Writing and examples

Keep prose concise and concrete. Prefer “For more details:” for supporting references. Link relevant API docs, guides, code, papers, model cards, and benchmarks; prefer release-pinned repository links. Verify URLs and claims rather than inventing them.

Use the simplest public API at the target revision, including necessary imports and setup. Run examples when practical and report verification limits. Put printed output in Python comments immediately after each corresponding print call, within the same code fence. Use the actual display format; representative output is acceptable if identified as illustrative or variable. For `TableTensor`, show size and semantic column blocks. Do not seed random examples solely to match illustrative output.

## Optional gist synchronization

Once a maintainer authorizes gist synchronization, update it after every draft edit, including small wording changes. Reuse the identified gist; when asked to update one without a link, locate the matching draft or clarify an ambiguous destination. Never assume an earlier release's gist ID.

Read the remote file before writing, reconcile divergent edits, and preserve other files, metadata, and visibility. Use structured payloads or file uploads for multiline content. With `gh gist edit`, supply the actual gist ID, remote filename, and working-copy path. Fetch the named file afterward and compare its content with the intended draft. After an uncertain result, read back before retrying; report failures without claiming synchronization.

Create gists, publish releases, change tags, post to issues or PRs, commit, or push only within the maintainer's authorization and repository policy. Report the draft location, any gist link and verified sync status, and material verification gaps briefly.

## Final checklist

- [ ] Stable minor target, preceding stable baseline, and release claims are verified.
- [ ] Announcement, linked contents, required sections, and variable highlight subsections follow the specified structure; Breaking changes appears only when needed.
- [ ] Contributor credits, migration steps, links, and public-API examples are checked; output comments follow each print call and verification gaps are disclosed.
- [ ] Requested removals stay removed, contents links match headings, and no personal paths, filenames, or gist IDs are assumed.
- [ ] The chosen destination is updated, any authorized gist sync is verified, and external actions remain within authorization.
