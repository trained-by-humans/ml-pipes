# Releasing

This guide covers release maintenance. Keep
[`CONTRIBUTING.md`](CONTRIBUTING.md) focused on local contributor setup and
test runs.

> [!IMPORTANT]
> Tagged releases are automated through GitHub Actions.
>
> This page covers local validation and dry-runs before tagging, not manual
> release steps.

## Release Flow

Push a matching `vX.Y.Z` tag and the workflow:

- rerun the shared CI smoke suite for the tagged commit
- validate the shared version and internal package pins
- build the release distributions
- publish to TestPyPI in dependency order
- publish to PyPI using trusted publishing

> [!IMPORTANT]
> Reruns only skip uploads when the existing index artifacts exactly match the
> current build. Missing files are still uploaded, and conflicting files for
> the same version fail the workflow.

Configure both the `testpypi` and `pypi` environments as trusted publishers
before the first public release. Once the workflow is in place, do not use
long-lived PyPI tokens.

If you want to check a release before pushing a tag, use the local steps
below.

## Release-Specific Setup

Use Python 3.11+ when possible. Under Python 3.10, the pinned requirements
below also install `tomli`.

Install the pinned release tooling:

```bash
python3 -m pip install -r .github/requirements-release.txt
```

## Validate Release Metadata

Before tagging, validate release metadata from the repository root:

```bash
python3 .github/scripts/release_packages.py --validate --tag v0.1.0
```

This checks published package versions, internal `ml-pipes` pins across
runtime dependencies and extras, the release tag, and package publish order.

## Local Dry-Run

Run a local dry-run:

```bash
python3 .github/scripts/release_packages.py --dry-run
```
