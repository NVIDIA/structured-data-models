# Continuous Integration

For installation and usage, see the [project README](../README.md).

## Pull request checks

By default, pull request CI runs lint, CPU tests, and documentation checks. You can trigger additional checks on a PR.

### Run GPU CI

A maintainer can post this comment on the PR, including on a draft. Replace `<SHA>` with the commit SHA to test.

```
/ok to test <SHA>
```

### Run NVSkills CI

A maintainer can post this comment on a PR with changes under `skills/` or `team-skills/`.

```
/nvskills-ci
```

### Run the full test suite

Add the `ci-full-test` label before the next PR update to enable additional CPU and GPU tests, plus GPU examples and benchmark smoke tests.

## PyPI releases

Publish a stable GitHub release for the intended `vX.Y.Z` tag. The **Publish** workflow builds and validates a wheel and source distribution from that tag, then uploads both through PyPI Trusted Publishing. Mark release candidates as prereleases to skip PyPI publishing.

Check the **Publish** run in **Actions** and confirm that the released version appears on [PyPI](https://pypi.org/project/structured-data-models/).
