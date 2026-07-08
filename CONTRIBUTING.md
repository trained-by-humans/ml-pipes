# Contributing

This guide only covers local contributor setup.

## Local Setup

`ml-pipes` is a multi-package repo. The repository root is a workspace, not an
installable package, so do not run `pip install -e .` from the root.

Create a virtual environment first:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip pytest
```

Install the packages you need in editable mode.

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

Add optional extras only when needed.

Inspection and otel:

```bash
python -m pip install -e "packages/core[inspection,otel]"
```

Torch:

```bash
python -m pip install -e packages/torch
```

The umbrella package under `packages/meta/` is only needed when you want to
check published install behavior. Most contributor work should install the
editable workspace packages directly.

## Run Tests

From the repository root:

```bash
python -m pytest
```

Install the packages that match the surfaces you are testing before running
the full suite. For example, many shared tests expect the common framework
stack (`core`, `tensor`, `vision`, and `onnx`) to be installed.

## IDE Setup

If your editor still shows unresolved imports after the editable install,
mark these directories as source roots and refresh indexing:

- `packages/core/src`
- `packages/tensor/src`
- `packages/vision/src`
- `packages/onnx/src`
- `packages/torch/src`
