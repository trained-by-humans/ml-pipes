# Releasing

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
3. Wait for the PyPI publish jobs and the final verification job to pass.

Do not rebuild artifacts locally and do not move the tag between staging and
promotion. Promotion is meant to publish the exact files that already passed
the staging flow.

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
- [`publish-package-to-index`](.github/workflows/publish-package-to-index.yml)
- [`verify-published-packages`](.github/workflows/verify-published-packages.yml)
- [`release-plan.toml`](release-plan.toml)
