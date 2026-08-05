# Release

This guide is for maintainers cutting an `ml-pipes` release. Keep
[`CONTRIBUTING.md`](CONTRIBUTING.md) focused on local development setup and
ordinary test runs.

Release automation already owns the package build and publish mechanics. This
page should explain the maintainer-facing release flow, the local preflight
checks, and the failure policy around reruns.

## Release Model

`ml-pipes` releases use a two-phase flow.

### 1. Stage The Release Candidate

Push a release tag such as `v0.1.0` or `v0.1.0rc1`. The
[`stage-release`](.github/workflows/stage-release.yml) workflow then:

- validates release metadata and publish order
- reruns the shared CI suite for the tagged commit
- builds all release artifacts
- writes a release artifact manifest
- uploads the artifacts to the GitHub release for the tag
- publishes packages to TestPyPI in dependency order
- verifies that the published packages install and import correctly

This phase is the only place where artifacts are built. The GitHub release for
the tag becomes the staged source of truth for promotion.

### 2. Promote The Staged Release

After the staged packages look correct on TestPyPI, run the
[`promote-release`](.github/workflows/promote-release.yml) workflow manually
for the same tag. The workflow:

- revalidates release metadata for the tag
- downloads the staged artifacts from the GitHub release
- verifies the downloaded files against the stored artifact manifest
- publishes packages to PyPI in dependency order
- verifies that the published packages install and import correctly

Promotion reuses the staged artifacts. It does not rebuild the release.

## Release Tooling Context

The release workflows behave similarly, but GitHub resolves different parts of
the tooling at different times.

- Workflow YAML, including same-repo reusable workflows under
  `.github/workflows/*.yml`, is fixed when the run starts.
- Helper scripts under `.github/scripts/*.py` come from whatever tree
  `actions/checkout` places in the job workspace.

That leads to one important difference between staging and promotion:

- `stage-release` starts from the tag push, so the workflow definition, helper
  scripts, and package code all line up with the tagged commit.
- `promote-release` is started manually, but the run still follows the branch
  or tag ref selected for that dispatch. Its workflow definition comes from
  that dispatched ref, while its helper scripts come from the tag after
  checkout.

Promotion can therefore mix newer workflow orchestration with older
tag-scoped helper scripts. That is workable when staging and promotion stay
close together, but the gap is real.

If staging and promotion drift apart, later workflow fixes, permission
changes, or verification-policy changes may apply during promotion even though
helper-script behavior still comes from the older staged tag.

Treat this as an operational constraint of the current design:

- keep staging and promotion close together
- do not assume a post-tag tooling fix applies to an already staged release
- if release tooling changes materially after staging, cut a new release
  candidate and restage instead of promoting under drifted tooling

For the GitHub Actions mechanics behind this, see:

- [Manually run a workflow](https://docs.github.com/en/actions/how-tos/manage-workflow-runs/manually-run-a-workflow)
- [Reuse workflows](https://docs.github.com/en/actions/sharing-automations/reusing-workflows)
- [actions/checkout](https://github.com/actions/checkout)

## Prerequisites

Before the first release, make sure repository automation has the required
credentials:

- `TEST_PYPI_API_TOKEN` for staging publishes
- `PYPI_API_TOKEN` for production publishes

Release order and published package membership come from
[`release-plan.toml`](release-plan.toml). Update that file together with any
change to the release package set.

For local preflight commands, install the pinned release tooling:

```bash
python3 -m pip install -r .github/requirements-release.txt
```

## Local Preflight

Run these checks from the repository root before tagging.

Validate version alignment, internal pins, and publish order:

```bash
python3 .github/scripts/validate_release_metadata.py --tag v0.1.0
```

Optionally build the release artifacts locally and check their metadata:

```bash
python3 .github/scripts/build_release_artifacts.py --outdir dist/release
python3 -m twine check dist/release/*
```

These local checks are for confidence before pushing the tag. The authoritative
release build still happens in GitHub Actions.

## Stage A Release

1. Make sure the target commit is ready to publish.
2. Run the local preflight checks.
3. Create and push the release tag.

```bash
git tag v0.1.0
git push origin v0.1.0
```

4. Watch the `stage-release` workflow complete.
5. Confirm that:
   - CI passed for the tagged commit
   - the GitHub release contains the built artifacts and manifest
   - TestPyPI publish and verification both succeeded

If you want additional manual confidence, use the staged TestPyPI packages or
the GitHub release artifacts after the workflow finishes.

## Promote To PyPI

1. Open the `promote-release` workflow in GitHub Actions.
2. Run it with the exact staged tag, such as `v0.1.0`.
3. Wait for the PyPI publish jobs to complete and review the final verification
   job.

Do not rebuild artifacts locally and do not move the tag between staging and
promotion. Promotion is meant to publish the exact files that already passed
the staging flow.

Published-package verification is advisory during promotion. A verification
warning still needs follow-up, but it does not mean the uploaded PyPI artifacts
were rolled back.

## Reruns And Failure Handling

Reruns are safe only when the release contents for the tag have not changed.

- If a package was not uploaded yet, a rerun can publish the missing files.
- If a package index already has files for that version with matching hashes,
  the publish workflow skips those files.
- If a package index already has files for that version with different hashes,
  the workflow fails. Treat that version as immutable and cut a new version
  instead of trying to replace files.

If the staged artifacts themselves need to change, bump the package version and
create a new release tag. Do not mutate an existing staged release in place.

## References

- [`stage-release`](.github/workflows/stage-release.yml)
- [`promote-release`](.github/workflows/promote-release.yml)
- [`publish-artifacts-to-index`](.github/workflows/publish-artifacts-to-index.yml)
- [`verify-published-packages`](.github/workflows/verify-published-packages.yml)
- [`release-plan.toml`](release-plan.toml)
