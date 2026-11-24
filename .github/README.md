# Continuous integration

For installation and usage, see the [project README](../README.md).

- **Default checks.** Pull requests run lint, CPU tests, and documentation checks. The copy PR bot automatically syncs ready-for-review PRs to `pull-request/<number>`, which triggers GPU tests.
- **`/ok to test <SHA>`.** A maintainer can comment on a PR with the commit SHA to trigger GPU tests for that revision, including on a draft PR.
- **`ci-full-test`.** Apply this label before the next PR update to enable additional CPU and GPU tests, plus GPU examples and benchmark smoke tests. Pushes to `main` run the full suites automatically.
- **`/nvskills-ci`.** NVSkills validation runs automatically for PR changes under `skills/` or `team-skills/`. A maintainer can comment `/nvskills-ci` on the PR to request another run.
- **Results.** See the PR's **Checks** tab or find **Test GPU** runs for `pull-request/<number>` in [Actions](https://github.com/NVIDIA/structured-data-models/actions).
- **Manual runs.** In **Actions**, use **Run workflow** to start **Lint**, **Test**, **Test GPU**, or **Documentation** on a selected branch. Open an existing run to rerun failed jobs.

## PyPI releases

Publish a stable GitHub release for the intended `vX.Y.Z` tag. The **Publish** workflow builds and validates a wheel and source distribution from that tag, then uploads both through PyPI Trusted Publishing. Mark release candidates as prereleases to skip PyPI publishing.

Check the **Publish** run in **Actions** and confirm that the released version appears on [PyPI](https://pypi.org/project/structured-data-models/).
