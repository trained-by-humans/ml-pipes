# ml-pipes

Umbrella meta-package for `ml-pipes` component installs.

`ml-pipes` is the install entrypoint for the project. It installs
`ml-pipes-core` by default and uses extras to pull in domain packages like
vision, ONNX, Torch, inspection, and OpenTelemetry support. The root
`ml_pipes` namespace remains a namespace package only, so public imports come
from component modules such as `ml_pipes.core`, `ml_pipes.tensor`,
`ml_pipes.vision`, `ml_pipes.onnx`, and `ml_pipes.torch`.

Example:

```bash
pip install 'ml-pipes[onnx,vision]'
```

This installs the ONNX and vision package chain while keeping imports
component-scoped:

```python
from ml_pipes.core import Pipeline
from ml_pipes.onnx import Infer
from ml_pipes.vision import Resize
```

Optional installs are requested through the umbrella package name. For
example, install OpenTelemetry support with `pip install 'ml-pipes[otel]'`,
not with nested extra syntax such as `ml-pipes[core[otel]]`.

For existing install profiles, the package matrix, and the public component
import model, see the
[package guide](../../docs/PACKAGES.md).
