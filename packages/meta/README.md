# ml-pipes

`ml-pipes` is the umbrella install package for the `ml-pipes` framework. It
installs `ml-pipes-core` by default and uses extras to pull in additional
framework packages.

After installation, import the component modules you use, such as
`ml_pipes.core`, `ml_pipes.tensor`, `ml_pipes.vision`, `ml_pipes.onnx`, or
`ml_pipes.torch`. See the main project docs and examples for the broader
framework usage model.

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
[package guide](https://github.com/trained-by-humans/ml-pipes/blob/main/docs/PACKAGES.md).
