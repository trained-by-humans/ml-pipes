# Contributing

This guide covers local contributor setup and a few repository-level
maintenance tasks.

## Local Setup

`ml-pipes` is a multi-package repo. The repository root is a workspace, not an
installable package, so do not run `pip install -e .` from the root.

Create a virtual environment first:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip pytest
```

Install the smallest editable package set that matches your work:

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

Torch without the full workspace setup:

```bash
python -m pip install -e packages/torch
```

Inspection and otel:

```bash
python -m pip install -e "packages/core[inspection,otel]"
```

The common shared-framework setup covers most contributor work and much of the
test suite. The umbrella package under `packages/meta/` is mainly for checking
published install behavior.

## Run Tests

From the repository root:

```bash
python -m pytest
```

Match your install profile to the surfaces you are testing. The common
shared-framework setup covers many shared tests; add `packages/torch` when
running Torch-specific tests.

## Release Workflow

Publish packages in dependency order:

1. `ml-pipes-core`
2. `ml-pipes-tensor`
3. `ml-pipes-vision`
4. `ml-pipes-onnx`
5. `ml-pipes-torch`
6. `ml-pipes`

Run a release dry-run from the repository root with:

```bash
python scripts/release_packages.py --dry-run
```

## IDE Setup

If your editor still shows unresolved imports after the editable install,
mark these directories as source roots and refresh indexing:

- `packages/core/src`
- `packages/tensor/src`
- `packages/vision/src`
- `packages/onnx/src`
- `packages/torch/src`
