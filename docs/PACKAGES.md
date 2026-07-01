# Packages

In `ml-pipes`, packages are the installable parts of the framework. The
framework uses a multi-package layout so the core pipeline harness can stay
small while domain packages carry their own dependency graph, docs, and
examples. That keeps heavy runtime integrations such as vision, ONNX, or
Torch optional instead of forcing every install to carry every dependency.

Each package therefore owns one coherent part of the framework surface, such
as the core harness, tensor operators, vision operators, ONNX integration, or
Torch integration. Those packages publish modules under the shared
`ml_pipes` namespace.

If you want to get a runnable example working first, start with
[examples/README.md](../examples/README.md). This page is the reference for
the published packages, primary install profiles, and public imports.

## Packaging Terms

- **Package**: an installable part of the framework such as `ml-pipes-vision`
- **Optional installs**: package-specific extras such as `inspection` or
  `otel` that add optional dependencies without creating new import paths
- **Depends on**: the direct dependencies a package needs, such as
  `ml-pipes-tensor`, `numpy`, or `opencv-python`
- **Public modules**: import surfaces such as `ml_pipes.vision`
- **Content**: the part of the framework the package carries, such as
  detection, segmentation, density, tracing, or benchmarking
- **Install profile**: an install shortcut such as `ml-pipes[vision]` or
  `ml-pipes-core[otel]`; profiles install packages or optional dependencies,
  but they do not create new import paths

## Install ml-pipes

`ml-pipes` is the installation entrypoint for the framework.

- `pip install ml-pipes` installs the core framework package, `ml-pipes-core`
- `pip install 'ml-pipes[vision]'` installs core together with the vision
  package chain
- `pip install 'ml-pipes[onnx,vision]'` installs core together with both the
  ONNX and vision package chains

That makes core the default shape of the framework, while umbrella profiles
add the extra package chains you want in the same install.

> [!NOTE]
> `ml-pipes` is the umbrella install package. It installs `ml-pipes-core` by
> default and exposes composed install profiles. For more on the umbrella
> package itself, see [`packages/meta/README.md`](../packages/meta/README.md).

## How To Use Packages

Once installed, use packages through their owning public modules.

For example, after installing `ml-pipes[onnx,vision]`:

```python
from ml_pipes.core import Pipeline
from ml_pipes.onnx import Infer
from ml_pipes.tensor import ArgMax
from ml_pipes.vision import Resize
```

The root `ml_pipes` namespace is intentionally only a shared namespace. Import
from the owning module shown in the package reference below, not from
top-level `ml_pipes`.

Package dependencies do not change ownership. For example, the ONNX package
depends on the tensor package, so an ONNX pipeline may import from both
`ml_pipes.onnx` and `ml_pipes.tensor`. The tensor operators still belong to
the tensor package, while the ONNX package owns only the ONNX runtime
surface.

## Package Reference

The table below is the package index for the framework packages. For
package-specific details, open the linked package README.

| Package                                         | Primary profile    | Depends on                                                          | Public modules                                                                                                                                                          | Content                                    |
|-------------------------------------------------|--------------------|---------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------|
| [ml-pipes-core](../packages/core/README.md)     | `ml-pipes`         | `numpy`, `typing_extensions`                                        | `ml_pipes.core`, `ml_pipes.standard`, `ml_pipes.validation`, `ml_pipes.tracing`, `ml_pipes.collectors`, `ml_pipes.factory`, `ml_pipes.benchmark`, `ml_pipes.inspection` | core pipeline framework and shared tooling |
| [ml-pipes-tensor](../packages/tensor/README.md) | `ml-pipes[tensor]` | `ml-pipes-core`, `numpy`                                            | `ml_pipes.tensor`                                                                                                                                                       | tensor data and tensor operators           |
| [ml-pipes-vision](../packages/vision/README.md) | `ml-pipes[vision]` | `ml-pipes-core`, `ml-pipes-tensor`, `numpy`, `opencv-python`        | `ml_pipes.vision`                                                                                                                                                       | vision payloads and vision tasks           |
| [ml-pipes-onnx](../packages/onnx/README.md)     | `ml-pipes[onnx]`   | `ml-pipes-core`, `ml-pipes-tensor`, `numpy`, `onnxruntime`          | `ml_pipes.onnx`                                                                                                                                                         | ONNX model and runtime integration         |
| [ml-pipes-torch](../packages/torch/README.md)   | `ml-pipes[torch]`  | `ml-pipes-core`, `ml-pipes-tensor`, `numpy`, `torch`, `torchvision` | `ml_pipes.torch`                                                                                                                                                        | Torch tensors and Torch integration        |

> [!NOTE]
> The table lists only primary profiles. For package-specific optional
> installs and profile details, check the owning package README.

> [!TIP]
> Profiles compose, so `ml-pipes[onnx,vision]` installs both package
> chains and supports imports from `ml_pipes.onnx`, `ml_pipes.tensor`,
> and `ml_pipes.vision`.

## Package Structure

Each workspace package follows the same basic layout:

```text
packages/<name>/
  pyproject.toml
  README.md
  src/
    ml_pipes/...
  LICENSE        # optional; only when this package needs different licensing
  examples/      # optional; package-local examples when they belong here
```

- `pyproject.toml` defines the published package metadata and dependencies.
- `README.md` explains what that package carries in framework terms.
- `src/ml_pipes/...` contains the code that package publishes into the shared
  `ml_pipes` namespace.
- `LICENSE` is only needed when a package must declare licensing different
  from the main project licensing.
- `examples/` is optional; use it when examples are truly package-specific
  rather than general framework examples under the repository-level
  [examples/README.md](../examples/README.md).
