## Architecture

### Pipeline

`Pipeline` is a list of callables executed in sequence. The output of each
step becomes the input of the next. Any Python callable can appear in the
list — operator instances, plain functions, or lambdas.

```python
from ml_pipes import Pipeline

pipeline = Pipeline([
    step_one,
    step_two,
    step_three,
])

result = pipeline(input_value)
```

When an operator accepts more than one positional argument, the pipeline
expects the current value to be a tuple matching the argument count. This is
how `Recall` injects a stored value alongside the flowing registry.

### Operators

Operators are the building blocks of a pipeline. They fall into four families:
**transform** (type changes), **tensor** (endomorphic on `TensorRegistry`),
**context** (side-channel), and **side-effect** (tap pattern).

The full operator reference — including all parameters, input/output types, and
the `as_` in-place/new-key contract — is in [OPERATORS.md](OPERATORS.md).

### Context

The pipeline is linear: a single value flows from step to step. Some
postprocessing steps need information computed much earlier — for example, the
resize transform from preprocessing is needed when projecting boxes back to
original image space. Threading this through every intermediate operator would
pollute all signatures.

The context system solves this with an immutable side-channel:

- `Store(name)` — saves the current value (or a tuple element via `index=`)
  into a context dictionary. The flowing value is unchanged.
- `Recall(name)` — retrieves a stored value and appends it to the current
  value, producing a tuple. The next operator then receives both.

```python
Resize((640, 640)),                      # current = (ImagePayload, ResizeTransform)
Store("resize_transform", index=1),      # store transform; current unchanged
Pick(0),                                 # current = ImagePayload
...                                      # inference and postprocessing
Recall("resize_transform"),              # current = (TensorRegistry, ResizeTransform)
ProjectBoxes(),                          # receives (registry, transform)
```

`Recall` is idempotent — the stored value is not consumed. Calling `Recall`
for the same key twice lets two operators (e.g. `ProjectMasks` and
`ProjectBoxes`) independently receive the same transform.

### TensorRegistry

`TensorRegistry` is the intermediate representation used throughout
postprocessing. It is a mutable named store of NumPy arrays, created by
`Extract` from the raw `RuntimeOutputs` of inference and passed through every
tensor operator until `ToDetections` or `ToSegmentations` converts it to a
typed output.

```python
from ml_pipes import TensorRegistry

registry = TensorRegistry({"boxes": boxes_array, "scores": scores_array})
registry["classes"] = classes_array
print(registry["boxes"].shape)
```

### Types

The types in `ml_pipes.types` describe the values flowing through the pipeline
at each stage:

| Type | Where it appears | Description |
|---|---|---|
| `ImagePayload` | after `Decode`, through preprocessing | HWC NumPy array with color space and layout metadata |
| `TensorPayload` | after `Normalize`, into `Infer` | CHW/NCHW float array with layout and dtype metadata |
| `RuntimeOutputs` | from `Infer` | raw ONNX output tensors with their graph names |
| `TensorRegistry` | from `Extract` through all postprocessing | named tensor store |
| `ResizeTransform` | stored via `Store`, recalled via `Recall` | scale, pad, and shape metadata from `Resize` |
| `Detections` | from `ToDetections` | typed output: boxes, scores, classes |
| `Segmentations` | from `ToSegmentations` | typed output: boxes, scores, classes, masks |
