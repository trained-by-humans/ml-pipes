# Packages

`ml-pipes` now ships as a small workspace of distributions that share the
same `ml_pipes` namespace package.

## Public Import Model

The root `ml_pipes` namespace is intentionally empty apart from submodules.
Import from the owning surface instead:

- `ml_pipes.core` for `Pipeline`, `Operator`, `Embed`, `Inline`, and context/runtime primitives
- `ml_pipes.standard` for core routing and region operators such as `Pick`, `Store`, `Recall`, `Batch`, and `Scatter`
- `ml_pipes.tensor` for tensor registry payloads and tensor operators
- `ml_pipes.vision` for payloads and operators in the vision domain
- `ml_pipes.onnx` for ONNX runtime integration
- `ml_pipes.torch` for Torch-domain operators
- `ml_pipes.validation`, `ml_pipes.tracing`, `ml_pipes.collectors`, `ml_pipes.factory`, `ml_pipes.benchmark`, and `ml_pipes.inspection` for the shared framework tooling modules

## Install Profiles

Published package installs:

```bash
pip install ml-pipes
pip install 'ml-pipes[vision]'
pip install 'ml-pipes[onnx,vision]'
pip install 'ml-pipes[torch,vision]'
pip install 'ml-pipes[inspection,onnx,vision]'
pip install 'ml-pipes[otel]'
```

From a repository checkout, install the workspace members you want to edit:

```bash
python -m pip install \
  -e packages/core \
  -e packages/tensor \
  -e packages/vision \
  -e packages/onnx \
  -e packages/torch \
  -e packages/meta
```

## Distribution Matrix

| Distribution | Public modules | Depends on | Notes |
|---|---|---|---|
| `ml-pipes-core` | `ml_pipes.core`, `ml_pipes.standard`, `ml_pipes.validation`, `ml_pipes.tracing`, `ml_pipes.collectors`, `ml_pipes.factory`, `ml_pipes.benchmark`, `ml_pipes.inspection` | `numpy`, `typing_extensions` | ships the framework harness and the standard operator family |
| `ml-pipes-tensor` | `ml_pipes.tensor` | `ml-pipes-core`, `numpy` | owns `TensorPayload`, `TensorRegistry`, and tensor-domain operators |
| `ml-pipes-vision` | `ml_pipes.vision` | `ml-pipes-core`, `ml-pipes-tensor`, `numpy`, `opencv-python` | owns image payloads, vision operators, density helpers, and tiling |
| `ml-pipes-onnx` | `ml_pipes.onnx` | `ml-pipes-core`, `ml-pipes-tensor`, `numpy`, `onnxruntime` | owns ONNX runtime invocation and output types |
| `ml-pipes-torch` | `ml_pipes.torch` | `ml-pipes-core`, `ml-pipes-tensor`, `numpy`, `torch`, `torchvision` | owns Torch tensors, Torch inference, and Torch tensor operators |
| `ml-pipes` | umbrella only | `ml-pipes-core` | adds extras so users can install stacks like `ml-pipes[vision]` or `ml-pipes[onnx,vision]` |

Core extras:

- `ml-pipes-core[inspection]` adds the inspection stack and the packages it formats by default
- `ml-pipes-core[otel]` adds the OpenTelemetry collector dependencies

Umbrella extras:

- `ml-pipes[vision]` installs `ml-pipes-vision` and its dependency chain
- `ml-pipes[onnx]` installs `ml-pipes-onnx`
- `ml-pipes[torch]` installs `ml-pipes-torch`
- `ml-pipes[inspection]` forwards to `ml-pipes-core[inspection]`
- `ml-pipes[otel]` forwards to `ml-pipes-core[otel]`

## Release Order

The package dependency graph requires this publish order:

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
