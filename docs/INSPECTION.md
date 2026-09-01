# Inspection

Inspection runs a pipeline once and captures the output at every operator
boundary. It is the tool for understanding what values actually flowed through
one call; use [TRACING.md](TRACING.md) when you need timings, failures, or
collector-driven observation instead.

## Quick Example

> [!NOTE]
> `Pipeline.inspect()` captures results without extra dependencies. To view or
> render them, install the `inspection` extra with the package chain you use,
> for example `ml-pipes[inspection,onnx,vision]`. Package modules register
> their specialized formatters when you import them.

Inspect a YOLOv8-style pipeline in the same way you run it: call `inspect()`
with the pipeline input, then view the captured result.

```python
from pathlib import Path

from ml_pipes.core import Pipeline
from ml_pipes.inspection import PipelineInspector
from ml_pipes.standard import Recall, Store
from ml_pipes.vision import Decode, DrawBoxes, LoadFile, SaveImage

pipeline = Pipeline(
    [
        LoadFile(),
        Decode(),
        Store("source_image"),
        # ... YOLOv8 preprocessing, inference, and postprocessing ...
        Recall("source_image", prepend=True),
        DrawBoxes(),
        SaveImage(Path("annotated.jpg"), at=0),
    ]
)

result = pipeline.inspect(<PATH/TO/IMAGE>)
PipelineInspector().show(result)
```

For a complete runnable inspection example, see
[`examples/run_inspect.py`](../examples/run_inspect.py).

![Inspection report overview](../.github/assets/yolo8_tiled_inspection_overview.png)

## How Inspection Works

`Pipeline.inspect(value)` works by tracing one pipeline execution. It follows
the same ordered span tree as tracing, with two additional conditions:

1. The output of every successful step is captured.
2. Each captured output is a snapshot, so later mutations to the live value do
   not change the `InspectionResult`.

`PipelineInspector` transforms those captured values into `StepView` and output
blocks. A renderer then turns the views into a presentation such as the built-in
HTML report. Inspection is observational: it does not alter pipeline execution
or its returned output.

```text
[input] -> Pipeline.inspect() -> [InspectionResult] -> PipelineInspector.build_views() -> [list[StepView]] -> Renderer.render() -> [HTML report]
```

> [!CAUTION]
> Inspection runs the complete pipeline and retains a snapshot of every step
> output. It therefore adds execution work and can consume substantial memory,
> especially for large images, tensors, batches, or nested regions. Use it for
> focused debugging and representative inputs rather than memory-sensitive or
> high-throughput production paths.

## When Inspection Is Useful

Use inspection when you want to:

- understand how the pipeline mutates data at each boundary
- identify the stage at which an unwanted change was introduced or the result
  began to drift from expectations
- double-check important stages to make sure they are working as expected
- share the awesome pipeline you have built with others

## Viewing An Inspection Result

`PipelineInspector.show()` displays the built-in HTML report inline in Jupyter
when available; otherwise it opens a browser report. Use it for an immediate,
interactive look at the result.

```python
from ml_pipes.inspection import PipelineInspector

PipelineInspector().show(result, orientation="horizontal")
```

### Save And Load A Captured Run

`InspectionResult` is captured data, separate from presentation. Persist it
when you want to inspect a run later or render it in a different environment.

```python
from ml_pipes.inspection import InspectionResult

# Save result
result.dump("inspection.pkl")

# Load and show result
saved_result = InspectionResult.load("inspection.pkl")
PipelineInspector().show(saved_result)
```

> [!CAUTION]
> Inspection results are serialized with Python pickle. Load only inspection
> files from sources you trust.

## Renderers

The built-in `HtmlRenderer` produces a self-contained browser report. The
default `PipelineInspector` uses it for `render()`, `save()`, and `show()`.

```python
inspector = PipelineInspector()

html = inspector.render(result, orientation="horizontal")
inspector.save(result, "inspection.html", orientation="horizontal")
```

For a different presentation, call `inspector.build_views(result)` and pass the
resulting `StepView` tree to an implementation of the `Renderer` protocol.

```python
from ml_pipes.inspection import Orientation, Renderer, StepView


class MyCustomRenderer(Renderer):
    def render(self, views: list[StepView], orientation: Orientation = "horizontal") -> str:
        return ...

views = inspector.build_views(result)
report = MyCustomRenderer().render(views)
```

## Formatters

Formatters translate captured values and operator steps into the output blocks
that renderers present. Value formatters describe values by type, while step
formatters can provide a specialized view for a particular operator.

Out of the box, inspection can render:

- primitive and otherwise unformatted values as text
- tuples and lists as ordered values or groups
- mappings and dataclasses as named groups
- Pydantic v1 and v2 models as groups of their declared public fields when
  Pydantic is installed
- nested structures, with recursive-reference protection and preview
  compaction

### Built-In Formatters

Core adds specialized formatting for:

- `bytes`, rendered with a size summary
- NumPy arrays, rendered with shape and dtype; three-channel HWC `uint8`
  arrays also receive an RGB image preview by default
- region operators, rendered with a region execution summary

Package modules add formatters for their own value types when imported. For
example, Vision renders `ImagePayload` using its explicit colour-space and
layout metadata.

### Adding Custom Formatters

Register a value formatter on a `PipelineInspector` instance to control how a
value type is represented in that inspector's output. The registration does
not change the pipeline or the captured `InspectionResult`.

```python
from ml_pipes.inspection import PipelineInspector, TextBlock

# Format values by type.
inspector = PipelineInspector().register_value_formatter(
    str,
    lambda value: [TextBlock("str", [("", value)])],
)
# Format a complete operator step.
inspector.register_step_formatter(MyOperator, format_my_operator_step)
```

> [!IMPORTANT]
> Registering a formatter for the same type twice raises an error by default.
> Set `override=True` only when you intentionally want to replace that
> inspector's existing formatter.

### Pydantic Models

When Pydantic is installed, inspection automatically registers a generic
formatter for `pydantic.BaseModel`. It renders declared model fields using the
v2 `model_fields` or v1 `__fields__` API; private attributes, extras, computed
fields, aliases, and arbitrary properties are not included. Register a
formatter for a concrete response type when a concise domain-specific summary
is more useful.

`pydantic_model_formatter()` creates the generic structural formatter. It
processes every declared field and nested value, while always detecting
recursive references. Set `max_depth` explicitly when inspection must bound
nesting traversal:

```python
from pydantic import BaseModel

from ml_pipes.inspection import PipelineInspector, pydantic_model_formatter

inspector = PipelineInspector().register_value_formatter(
    BaseModel,
    pydantic_model_formatter(max_depth=4),
)
```

`max_depth` accepts a positive integer or `None`; `None` leaves nesting
unbounded.

If a captured Pydantic model is encountered while Pydantic itself is
unavailable, inspection emits `InspectionWarning` once for that model type and
uses the normal text representation instead.

### Raw Image Arrays

To make reports easier to scan, inspection detects image-like outputs and
renders an image preview. Raw arrays can have image-shaped dimensions while
lacking the information needed to interpret them safely, such as channel order
or color space. Register an explicit convention when your application uses raw
image arrays.

By default, inspection recognizes an HWC `uint8` array with three channels and
interprets its channels as RGB. If your raw images use OpenCV's BGR convention,
replace the ndarray formatter on the inspector with a BGR configuration:

```python
import numpy as np
from ml_pipes.inspection import PipelineInspector, ndarray_image_formatter

inspector = PipelineInspector().register_value_formatter(
    np.ndarray,
    ndarray_image_formatter(default_color_space="BGR"),
)
```

The formatter supports `"RGB"` and `"BGR"`. Keep the built-in formatter, or
register `ndarray_image_formatter(default_color_space="RGB")`, when RGB is
your application's default; use the BGR configuration above when it differs.

> [!NOTE]
> Prefer `ImagePayload` from the Vision package for image boundaries. It keeps
> the image array together with explicit colour-space and layout metadata, so
> inspection can render it accurately without this override.
