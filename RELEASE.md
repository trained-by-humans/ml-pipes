# Releasing

This guide covers repository release maintenance tasks. Keep
[`CONTRIBUTING.md`](CONTRIBUTING.md) focused on everyday contributor setup
and test runs.

## Release-Specific Setup

Use a Python 3.11+ environment for local release validation when possible.
If you run release validation under Python 3.10, install `tomli` because the
standard-library `tomllib` module is not available there yet.

Install the local release tooling you need:

```bash
python -m pip install -U build hatchling twine
```

If you are using Python 3.10, also install:

```bash
python -m pip install -U tomli
```

## Version Alignment

All published package versions must stay aligned. Before cutting a release,
confirm that every `packages/*/pyproject.toml` version matches the intended
release tag `vX.Y.Z`.

Publish packages in dependency order:

1. `ml-pipes-core`
2. `ml-pipes-tensor`
3. `ml-pipes-vision`
4. `ml-pipes-onnx`
5. `ml-pipes-torch`
6. `ml-pipes`

## Validate Release Metadata

Validate release metadata from the repository root with:

```bash
python scripts/release_packages.py --validate --tag v0.1.0
```

This check is strict. If the command cannot read package `pyproject.toml`
files, it fails and tells you to use Python 3.11+ or install `tomli`.

## Local Dry-Run

Run a release dry-run from the repository root with:

```bash
python scripts/release_packages.py --dry-run
```

## GitHub Release Flow

Publishing happens in GitHub Actions after you push a matching `vX.Y.Z` tag.
The release workflow validates the shared version, builds all six
distributions, publishes them to TestPyPI in dependency order, then publishes
them to PyPI using trusted publishing.

The repository should configure both the `testpypi` and `pypi` environments
as trusted publishers before the first public release. Do not upload with
long-lived PyPI tokens once the workflow is in place.
