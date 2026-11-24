# Continuous Integration

For installation and usage, see the [project README](../README.md).

## Pull request checks

By default, pull request CI runs lint, CPU tests, and documentation checks. You can trigger additional checks on a PR.

### Run the full test suite

A maintainer can label a PR with `ci-full-test` to enable additional CPU and GPU test cases, GPU examples, and benchmark smoke tests on the next PR update.

Replace `<PR Number>` with the PR number.

```sh
gh pr edit "<PR Number>" --add-label ci-full-test
```

### Run GPU CI

A maintainer can post this comment on the PR, including on a draft. Replace `<SHA>` with the commit SHA to test.

```
/ok to test <SHA>
```

### Run NVSkills CI

A maintainer can post this comment on a PR with changes under `skills/`.

```
/nvskills-ci
```

## PyPI releases

Publish a stable GitHub release for the intended `vX.Y.Z` tag. The **Publish** workflow builds and validates a wheel and source distribution from that tag, then uploads both through PyPI Trusted Publishing. Mark release candidates as prereleases to skip PyPI publishing.

Check the **Publish** run in **Actions** and confirm that the released version appears on [PyPI](https://pypi.org/project/structured-data-models/).
