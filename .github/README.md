# Continuous Integration

## Trigger additional checks on pull requests

By default, pull request CI runs lint, CPU tests, and documentation checks.

### 1. Run the full test suite

A maintainer can label a PR with `ci-full-test` to enable additional CPU and GPU test cases, GPU examples, and benchmark smoke tests. Apply the label before triggering CI.

Replace `<PR Number>` with the PR number.

```sh
gh pr edit "<PR Number>" --add-label ci-full-test
```

If CI already read the labels, trigger a new [Test](workflows/test.yml) run by pushing a commit or closing and reopening the PR. For [Test GPU](workflows/test-gpu.yml), select **Re-run all jobs** on the run in GitHub Actions to check the label again. You can also use the run ID from the Actions URL.

```sh
gh run rerun "<Run ID>"
```

### 2. Run GPU CI

A maintainer can post this comment on a PR, including a draft. Replace `<SHA>` with the commit SHA to test.

```
/ok to test <SHA>
```

### 3. Run NVSkills CI

A maintainer can post this comment on a PR with changes under `skills/`.

```
/nvskills-ci
```

## PyPI releases

Publish a stable GitHub release for the intended `vX.Y.Z` tag. The [Publish](workflows/publish.yml) workflow builds and validates a wheel and source distribution from that tag, then uploads both through PyPI Trusted Publishing. Mark release candidates as prereleases to skip PyPI publishing.

Check the **Publish** run in **Actions** and confirm that the released version appears on [PyPI](https://pypi.org/project/structured-data-models/).
