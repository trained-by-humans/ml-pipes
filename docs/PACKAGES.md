# Packages

This page is the reference for published packages, primary install profiles,
and public imports.

## Why Packages
`ml-pipes` uses a multi-package layout so the core pipeline harness can stay
small and generic while heavier domains such as vision, ONNX, and Torch stay
optional.

Each package owns one coherent part of the framework surface and publishes it
under the shared `ml_pipes` namespace.

## Packaging Terms

- **Package**: an installable part of the framework such as `ml-pipes-vision`
- **Optional installs**: package-specific extras such as `inspection` or
  `otel` that add optional dependencies without creating new import paths
- **Install profile**: an install shortcut such as `ml-pipes[vision]` or
  `ml-pipes-core[otel]`; profiles install packages or optional dependencies,
  but they do not create new import paths
- **Depends on**: the direct `ml-pipes` package dependencies a package needs,
  such as `ml-pipes-core` or `ml-pipes-tensor`
- **Public modules**: import surfaces such as `ml_pipes.vision`
- **Content**: the part of the framework the package carries, such as
  detection, segmentation, density, tracing, or benchmarking

## Install ml-pipes

`ml-pipes` is the umbrella install package. `pip install ml-pipes` installs
`ml-pipes-core`, and extras add optional package chains.

- `pip install ml-pipes` installs the core framework package, `ml-pipes-core`
- `pip install 'ml-pipes[vision]'` installs core together with the vision
  package chain
- `pip install 'ml-pipes[onnx,vision]'` installs core together with both the
  ONNX and vision package chains

That makes core the default shape of the framework, while umbrella profiles
add the extra package chains you want in the same install.

The `inspection` profile installs the shared inspection renderers from core.
Combine it with package profiles such as `ml-pipes[inspection,tensor]` or
`ml-pipes[inspection,onnx,vision]` for package-specific inspection formatting.
That formatting follows the package modules you import, not every package that
could be installed.

> [!NOTE]
> For more on the umbrella package itself, see
> [`packages/meta/README.md`](../packages/meta/README.md).

## How To Use Packages

Once installed, import from the owning public module shown in the package
reference below, not from top-level `ml_pipes`.

For example, after installing `ml-pipes[onnx,vision]`:

```python
from ml_pipes.core import Pipeline
from ml_pipes.onnx import Infer
from ml_pipes.tensor import ArgMax
from ml_pipes.vision import Resize
```

Package dependencies do not change ownership. For example, an ONNX pipeline
may import from both `ml_pipes.onnx` and `ml_pipes.tensor`: ONNX owns the
runtime boundary, while tensor owns the shared tensor postprocess surface.

## Package Reference

Package names indicate ownership, not full domain completeness. The table
below lists the current surface each package delivers. For package-specific
details, open the linked package README.

| Package                                         | Primary profile    | Depends on                                                          | Public modules                                                                                                                                                          | Delivers                                                                              |
|-------------------------------------------------|--------------------|---------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------|
| [ml-pipes-core](../packages/core/README.md)     | `ml-pipes`         | `—`                                                                 | `ml_pipes.core`, `ml_pipes.standard`, `ml_pipes.validation`, `ml_pipes.tracing`, `ml_pipes.collectors`, `ml_pipes.factory`, `ml_pipes.benchmark`, `ml_pipes.inspection` | pipeline composition, generic data flow, and framework tooling                        |
| [ml-pipes-tensor](../packages/tensor/README.md) | `ml-pipes[tensor]` | `ml-pipes-core`                                                     | `ml_pipes.tensor`                                                                                                                                                       | shared NumPy-side tensor handling and reusable tensor postprocess                     |
| [ml-pipes-vision](../packages/vision/README.md) | `ml-pipes[vision]` | `ml-pipes-core`, `ml-pipes-tensor`                                  | `ml_pipes.vision`                                                                                                                                                       | image preparation, typed vision results, tiling, and visualization                    |
| [ml-pipes-onnx](../packages/onnx/README.md)     | `ml-pipes[onnx]`   | `ml-pipes-core`, `ml-pipes-tensor`                                  | `ml_pipes.onnx`                                                                                                                                                         | ONNX Runtime inference boundary and output handoff                                    |
| [ml-pipes-torch](../packages/torch/README.md)   | `ml-pipes[torch]`  | `ml-pipes-core`, `ml-pipes-tensor`                                  | `ml_pipes.torch`                                                                                                                                                        | Torch execution stages, explicit NumPy/Torch crossing, and on-device postprocess      |

> [!NOTE]
> The table lists only primary profiles. For package-specific optional
> installs and profile details, check the owning package README.

> [!TIP]
> Profiles compose, so `ml-pipes[onnx,vision]` installs both package
> chains and supports imports from `ml_pipes.onnx`, `ml_pipes.tensor`,
> and `ml_pipes.vision`.

## How Packages Usually Connect

Packages usually connect stage-by-stage through data boundaries. Some common
handoffs are:

| From   | To      | Connection                                                              | Note                                                                                         |
|--------|---------|-------------------------------------------------------------------------|----------------------------------------------------------------------------------------------|
| Vision | ONNX    | `TensorPayload -> RuntimeOutputs`                                       | `Normalize()` prepares model input, then `Infer()` runs ONNX Runtime                         |
| ONNX   | Tensor  | `RuntimeOutputs -> TensorRegistry`                                      | `Extract()` pulls named outputs into shared postprocess                                      |
| Tensor | Vision  | `TensorRegistry -> TensorRegistry` | Vision operators postprocess, render, or log named prediction tensors in the registry |
| Tensor | Torch   | `TensorPayload -> TorchTensorPayload`                                   | `ToTorch()` crosses a `TensorPayload` into the Torch domain                                  |
| Torch  | Tensor  | `TorchTensorRegistry -> TensorRegistry`                                 | `ToNumpyRegistry()` hands Torch results back to NumPy-side packages                          |

## Package Structure

Each workspace package follows the same basic layout:

```text
packages/<name>/
  pyproject.toml
  README.md
  docs/           # package-owned guides and deep dives
  src/
    ml_pipes/...
  LICENSE         # package-local license file shipped with the distribution
  examples/      # optional; package-local examples when they belong here
```

- `pyproject.toml` defines the published package metadata and dependencies.
- `README.md` explains what that package carries in framework terms.
- `docs/` holds package-specific guides that do not belong in the shared
  framework docs under the repository-level `docs/`.
- `src/ml_pipes/...` contains the code that package publishes into the shared
  `ml_pipes` namespace.
- `LICENSE` is the package-local license file shipped with that distribution
  and referenced by the published package metadata; it may carry the
  repository-wide project license text or package-specific license terms.
- `examples/` is optional; use it when examples are truly package-specific
  rather than general framework examples under the repository-level
  [examples/README.md](../examples/README.md).
