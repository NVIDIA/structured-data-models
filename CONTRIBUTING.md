# Contributing to structured-data-models

If you are interested in contributing to structured-data-models, your contributions will fall
into three categories:

1. You want to report a bug, feature request, or documentation issue
   - File an [issue](https://github.com/NVIDIA/structured-data-models/issues/new/choose)
     describing what you encountered or what you want to see changed.
   - Please run and paste the output of `python -m torch.utils.collect_env` while reporting a bug to gather and report relevant environment details.
   - The project maintainers will evaluate the issues and triage them, scheduling them for a release. If you believe the issue needs priority attention, comment on the issue to notify the team.
2. You want to propose a new Feature and implement it
   - Post about your intended feature, and we shall discuss the design and
     implementation.
   - Once we agree that the plan looks good, go ahead and implement it, using
     the [code contributions](#code-contributions) guide below.
3. You want to implement a feature or bug-fix for an outstanding issue
   - Follow the [code contributions](#code-contributions) guide below.
   - If you need more context on a particular issue, please ask and we shall
     provide.

## Code contributions

### Your first issue

1. Read the project's [README.md](https://github.com/NVIDIA/structured-data-models/blob/main/README.md)
   to learn how to install the package.
2. Find an issue to work on. The best way is to look for the [good first issue](https://github.com/NVIDIA/structured-data-models/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)
   or [help wanted](https://github.com/NVIDIA/structured-data-models/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22) labels
3. Comment on the issue saying you are going to work on it.
4. Get familiar with the [project guidelines](AGENTS.md).
5. Code! Make sure to update unit tests!
6. When done, [create your pull request](https://github.com/NVIDIA/structured-data-models/compare?template=default.md) and fill in the template it loads.
7. Verify that CI passes all [status checks](https://help.github.com/articles/about-status-checks/), or fix if needed.
8. Wait for other developers to review your code and update code as needed.
9. Once reviewed and approved, a project maintainer will merge your pull request.

Remember, if you are unsure about anything, don't hesitate to comment on issues and ask for clarifications!

### Signing Off Your Work

- We require that all contributors "sign-off" on their commits. This certifies that the contribution is your original work, or you have rights to submit it under the same license, or a compatible license.

  - Any contribution which contains commits that are not Signed-Off will not be accepted.

- To sign off on a commit you simply use the `--signoff` (or `-s`) option when committing your changes:

  ```bash
  $ git commit -s -m "Add cool feature."
  ```

  This will append the following to your commit message:

  ```
  Signed-off-by: Your Name <your@email.com>
  ```

- Full text of the DCO (https://developercertificate.org/):

  ```
    Developer Certificate of Origin
    Version 1.1

    Copyright (C) 2004, 2006 The Linux Foundation and its contributors.

    Everyone is permitted to copy and distribute verbatim copies of this
    license document, but changing it is not allowed.


    Developer's Certificate of Origin 1.1

    By making a contribution to this project, I certify that:

    (a) The contribution was created in whole or in part by me and I
        have the right to submit it under the open source license
        indicated in the file; or

    (b) The contribution is based upon previous work that, to the best
        of my knowledge, is covered under an appropriate open source
        license and I have the right under that license to submit that
        work with modifications, whether created in whole or in part
        by me, under the same open source license (unless I am
        permitted to submit under a different license), as indicated
        in the file; or

    (c) The contribution was provided directly to me by some other
        person who certified (a), (b) or (c) and I have not modified
        it.

    (d) I understand and agree that this project and the contribution
        are public and that a record of the contribution (including all
        personal information I submit with it, including my sign-off) is
        maintained indefinitely and may be redistributed consistent with
        this project or the open source license(s) involved.
  ```

### Managing PR labels

Each PR must be labeled according to whether it is a "breaking" or "non-breaking" change (using GitHub labels). This is used to highlight changes that users should know about when upgrading.

For structured-data-models, a "breaking" change is one that modifies the public, non-experimental, Python API in a
non-backward-compatible way. Backward-compatible API changes to the Python
API (such as adding a new keyword argument to a function) do not need to be labeled.

Additional labels must be applied to indicate whether the change is a feature, improvement, bugfix, or documentation change. See the shared RAPIDS documentation for these labels: https://github.com/NVIDIA/kb/issues/42.

### Seasoned developers

Once you have gotten your feet wet and are more comfortable with the code, you
can look at the prioritized issues of our next release in our [project board](https://github.com/orgs/NVIDIA/projects/80).

> **Pro Tip:** Always look at the release board with the highest number for
> issues to work on. This is where RAPIDS developers also focus their efforts.

Look at the unassigned issues, and find an issue you are comfortable with
contributing to. Start with _Step 3_ from above, commenting on the issue to let
others know you are working on it. If you have any questions related to the
implementation of the issue, ask them in the issue instead of the PR.

### Branches and Versions

Development targets `main` by default. Maintainers may request a `release/X.Y` branch for release-specific changes.

### Additional details

For all development, your changes should be pushed into a branch (created using the naming instructions below) in your own fork of structured-data-models and then create a pull request when the code is ready.

### Branch naming

Branches used to create PRs should have a name of the form `<type>-<name>`
which conforms to the following conventions:

- Type:
  - fea - For if the branch is for a new feature(s)
  - enh - For if the branch is an enhancement of an existing feature(s)
  - bug - For if the branch is for fixing a bug(s) or regression(s)
- Name:
  - A name to convey what is being worked on
  - Please use dashes or underscores between words as opposed to spaces.

## Attribution

Portions adopted from https://github.com/pytorch/pytorch/blob/main/CONTRIBUTING.md under the BSD 3-Clause License distributed in [`third_party/pytorch/LICENSE`](third_party/pytorch/LICENSE).
