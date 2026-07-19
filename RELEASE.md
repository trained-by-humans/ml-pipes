# Releasing

This guide covers repository release maintenance tasks. Keep
[`CONTRIBUTING.md`](CONTRIBUTING.md) focused on everyday contributor setup
and test runs.

## Release-Specific Setup

Use a Python 3.11+ environment for local release validation when possible.
If you run release validation under Python 3.10, install `tomli` because the
standard-library `tomllib` module is not available there yet.

Install the pinned local release tooling used by the release workflow:

```bash
python3 -m pip install -r requirements-release.txt
```

If you are using Python 3.10, also install:

```bash
python3 -m pip install -U tomli
```

## Validate Release Metadata

Before cutting a release, validate release metadata from the repository root:

```bash
python3 .github/scripts/release_packages.py --validate --tag v0.1.0
```

This checks that all published package versions stay aligned, internal
`ml-pipes` pins stay aligned across runtime dependencies and extras, and the
release tag matches that shared version.

Publish packages in dependency order. `.github/scripts/release_packages.py`
validates the current publish order before building or publishing.

## Local Dry-Run

Run a release dry-run from the repository root with:

```bash
python3 .github/scripts/release_packages.py --dry-run
```

This command builds the distributions locally and requires the release
tooling from the setup step above.

## Release Flow

Publishing happens in GitHub Actions after you push a matching `vX.Y.Z` tag.
The release workflow runs this sequence:

- rerun the shared CI smoke suite for the tagged commit
- validate the shared version and internal package pins
- build the release distributions
- publish to TestPyPI in dependency order
- publish to PyPI using trusted publishing

> [!IMPORTANT]
> Reruns only skip uploads when the existing index artifacts exactly match the
> current build. Missing files are still uploaded, and conflicting files for
> the same version fail the workflow.

The repository should configure both the `testpypi` and `pypi` environments
as trusted publishers before the first public release. Do not upload with
long-lived PyPI tokens once the workflow is in place.
