# Contributing

This guide covers local contributor setup and a few repository-level
maintenance tasks.

## Local Setup

`ml-pipes` is a multi-package repo. The repository root is a workspace, not an
installable package, so do not run `pip install -e .` from the root.

Contributor installs are local workspace installs. The package
`pyproject.toml` files describe the published consumer dependency graph, so
from a fresh clone do not rely on standalone editable installs for packages
that depend on unpublished siblings.

Create a virtual environment first:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip pytest
```

Install the smallest local package set that matches your work:

Core-only work:

```bash
python -m pip install -e packages/core
```

Common shared-framework setup:

```bash
python -m pip install \
  -e packages/core \
  -e packages/tensor \
  -e packages/vision \
  -e packages/onnx
```

Full workspace setup:

```bash
python -m pip install \
  -e packages/core \
  -e packages/tensor \
  -e packages/vision \
  -e packages/onnx \
  -e packages/torch \
  -e packages/meta
```

Add optional surfaces only when needed:

Torch contributor setup:

```bash
python -m pip install \
  -e packages/core \
  -e packages/tensor \
  -e packages/torch
```

Inspection and otel contributor setup:

```bash
python -m pip install \
  -e "packages/core[inspection,otel]" \
  -e packages/tensor \
  -e packages/vision \
  -e packages/onnx
```

The common shared-framework setup covers most contributor work and much of the
test suite. The umbrella package under `packages/meta/` is mainly for checking
published install behavior. Once the packages are published, consumer installs
should use the published package names and profiles from the package docs
rather than these local workspace commands.

## Run Tests

From the repository root:

```bash
python -m pytest
```

Match your install profile to the surfaces you are testing. The common
shared-framework setup covers many shared tests; add `packages/torch` when
running Torch-specific tests.

## Release Workflow

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

Validate release metadata from the repository root with:

```bash
python scripts/release_packages.py --validate --tag v0.1.0
```

Run a release dry-run from the repository root with:

```bash
python scripts/release_packages.py --dry-run
```

Publishing happens in GitHub Actions after you push a matching `vX.Y.Z` tag.
The release workflow validates the shared version, builds all six
distributions, publishes them to TestPyPI in dependency order, then publishes
them to PyPI using trusted publishing.

The repository should configure both the `testpypi` and `pypi` environments
as trusted publishers before the first public release. Do not upload with
long-lived PyPI tokens once the workflow is in place.

## IDE Setup

If your editor still shows unresolved imports after the editable install,
mark these directories as source roots and refresh indexing:

- `packages/core/src`
- `packages/tensor/src`
- `packages/vision/src`
- `packages/onnx/src`
- `packages/torch/src`
