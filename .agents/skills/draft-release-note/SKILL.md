---
name: draft-release-note
description: Draft and revise SDM minor-release notes for vX.Y.0 from repository changes, with highlights, breaking changes, contributor credits, and examples. Use for minor releases only, not patch or major releases.
---

# Draft Release Note

Draft researcher-facing SDM release notes for stable minor releases (`vX.Y.0`, with a nonzero minor version). Clarify scope for patch, major, or prerelease requests.

## Establish the scope

Read repository instructions and preserve existing draft edits. Use the maintainer's chosen destination and naming conventions.

Identify the target tag or agreed commit and the preceding stable release, including its latest patch. Inspect the comparison, merged PRs, public APIs, examples, and documentation. Ground claims in that range. Verify each model's task, provenance, and whether it ships in SDM.

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

- **Highlights (required):** Lead with the most significant new researcher capability. Favor new models/tasks, usable training/adaptation workflows, substantial public APIs, and measured performance improvements. A standalone highlight should name a concrete capability researchers can try through a public entry point. Judge impact, not commit volume; group related additions into themes, with per-model bullets explaining differences. Fold supporting fixes, refactors, and guide updates into those themes or omit them. Do not add a generic Features section or inventory of minor additions.
- **Breaking changes (if any):** Include removed/renamed APIs and changed defaults or behavior that alter existing results. In one or two sentences per migration, name the public API, old-to-new usage (including restoring old behavior), and PR; omit internal adjustments.
- **New contributors (required):** List first-time contributors by GitHub handle with links to their first merged PRs. Verify first-time status against earlier history or GitHub metadata. If none, say so; if verification is blocked, retain the section and report the gap rather than inventing credits.
- **Full changelog (required):** Link the exact baseline-to-target GitHub comparison.

## Writing and examples

Explain the task and benefit before interface details. If a maintainer-written draft exists for this release, keep its leading themes, example purpose, and resources; add worthwhile, well-supported highlights. Earlier-release references guide style and emphasis, not content. Keep verified migrations and credits, flagging differences in the handoff. Treat references as read-only unless edits are requested.

Link each new model's public API, paper/blog, upstream code, and weights inline when available; note renames of previously published models. Prefer rendered docs and their images; use release-pinned repository links for training code, examples, benchmarks, or resources without rendered pages. Use researcher resources rather than PR history to explain highlights. Put remaining guides and evaluation scripts in a short “For more details:” list. Reuse helpful documentation figures with descriptive alt text and available light/dark variants. Verify URLs and claims.

Code is optional; a figure or model list may suffice. Choose examples that directly demonstrate the headline capability and meaningful output, not adjacent low-level components. For training/building-block releases, show the distinctive reusable component and link the full training guide; avoid long launch commands. Use the simplest public API at the target revision, with imports and setup. Run examples when practical and report limits. Put printed output in Python comments immediately after each print call in the same fence, using the real display format. Identify representative output as illustrative or variable; do not seed randomness solely to match it.

Briefly report the draft location and any material verification gaps.

## Final checklist

- [ ] Minor-release target, stable baseline, and claims are verified.
- [ ] Required structure, variable highlight count, and conditional Breaking changes are correct.
- [ ] Credits, migrations, links, examples, and output comments are checked; gaps are disclosed.
- [ ] Latest edits and removals are preserved, contents links match headings, and no personal defaults are assumed.
- [ ] The draft is available at the chosen destination.
