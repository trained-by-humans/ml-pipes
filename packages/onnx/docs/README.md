# ONNX Runtime

`ml_pipes.onnx` keeps ONNX Runtime invocation as one explicit pipeline boundary inside `ml-pipes`.

For the full package surface and operator signatures, see [`INDEX.md`](./INDEX.md).

## Package Profile

| Dimension       | Classification                                                                   |
|-----------------|----------------------------------------------------------------------------------|
| Role / Function | `Inference (Scaffold)`                                                           |
| Task Type       | General (any tensor-based task)                                                   |
| Data Type       | `Tensors`                                                                        |

## Scope And Use Cases

This package is intentionally thin. It is not a general ONNX tooling layer for
export, graph editing, or training. It owns the runtime call and the
small value model around that call.

**So use this package when a pipeline needs to run an exported ONNX model as one explicit stage.**

The package is responsible for:

- running an exported ONNX model
- validate layout and optional dtype expectations at the runtime boundary
- provide transition to postprocessing by extracting named tensors into `TensorRegistry`
- support batch operation by splitting one batched runtime call back into per-sample outputs

> [!NOTE]
> ONNX does not own preprocessing, generic tensor postprocess, or task-specific
> results.

## Design Principles

- Keep runtime invocation as one visible operator.
- Keep the runtime boundary small and explicit: `TensorPayload` in,
  `RuntimeOutputs` out.
- Make provider choice, layout expectations, optional dtype checks, and
  serialization explicit.
- Keep model outputs raw until another package interprets them.
- Hand off tensor math to `ml_pipes.tensor` and task semantics to the owning
  task package.

## Where ONNX Fits

ONNX is the runtime stage in the middle of the pipeline. Upstream packages
prepare a `TensorPayload`; downstream packages extract or distribute raw
outputs before continuing with Tensor-domain or task-specific postprocess.

At a high level, a common flow looks like this:

```text
┌──────────────────────────────────────────────────────────┐
│ Input / Task Domain                                      │
├─ LoadFile -> Decode -> Resize -> Normalize -> ...        │
└────────┬─────────────────────────────────────────────────┘
         |
         | TensorPayload
         ▼
┌──────────────────────────────────────────────────────────┐
│ ONNX Domain                                              │
├─ Infer                                                   │
└────────┬─────────────────────────────────────────────────┘
         |
         | RuntimeOutputs
         ▼
┌──────────────────────────────────────────────────────────┐
│ Tensor Domain                                            │
├─ Extract / Distribute -> Slice -> ArgMax -> ...          │
└────────┬─────────────────────────────────────────────────┘
         |
         | TensorRegistry
         ▼
┌──────────────────────────────────────────────────────────┐
│ Specific Task Domain (Vision, Language, etc.)             │
└──────────────────────────────────────────────────────────┘
```

That split keeps the runtime call easy to see, replace, and benchmark without
mixing it with postprocess or visualization.

## Using ONNX In Pipelines

In practice, ONNX usually appears in one of two pipeline shapes: single-sample
inference, or batched inference followed by `Distribute()` back into
per-sample flow.

### Single-Sample Inference

Use this shape when one pipeline run should correspond to one input sample and
one downstream result flow. Upstream preparation produces one
`TensorPayload`, `Infer(...)` runs once, and the extracted outputs continue
through the normal Tensor and task-specific postprocess.

```python
from ml_pipes.core import Pipeline
from ml_pipes.onnx import Extract, Infer
from ml_pipes.standard import Pick, Recall, Store
from ml_pipes.tensor import ArgMax, GatherScores, Slice, Squeeze, Transpose
from ml_pipes.vision import ConvertBoxFormat, Decode, LoadFile, NMS, Normalize, ProjectBoxes, Resize

pipeline = Pipeline([
    LoadFile(),
    Decode(),
    Resize((640, 640)),
    Store("resize_transform", source=1),
    Pick(0),
    Normalize(),
    Infer(model_path),
    Extract("output0", as_="preds"),
    Squeeze("preds"),
    Transpose("preds"),
    Slice("preds", slice(None, 4), as_="boxes"),
    Slice("preds", slice(4, None), as_="scores"),
    ArgMax("scores", as_="classes"),
    GatherScores("scores", "classes"),
    ConvertBoxFormat(from_="cxcywh"),
    NMS(),
    Recall("resize_transform"),
    ProjectBoxes(),
])
```

This is the shape used in
[`examples/run_yolo8_onnx.py`](../../../examples/run_yolo8_onnx.py).

### Batched Inference

Use this shape when batching improves throughput but the rest of the pipeline
should still continue as one result flow per original sample. Upstream steps
prepare multiple payloads, `Collate()` assembles them into one batched
payload, `Infer(...)` runs once over the batch, and `Distribute()` hands each
sample back to the downstream pipeline after the shared runtime step.

```python
from ml_pipes.core import Pipeline
from ml_pipes.onnx import Distribute, Extract, Infer
from ml_pipes.standard import Batch, UnBatch
from ml_pipes.tensor import Collate

pipeline = Pipeline([
    ...,
    Batch(size=4, timeout=0.05),
    Collate(),
    Infer(model_path, serialize=True),
    Distribute(),
    UnBatch(),
    Extract("output0", as_="preds"),
    ...,
])
```

This shape appears in
[`examples/run_yolo8_batch.py`](../../../examples/run_yolo8_batch.py).

## Providers And Concurrency

`Infer(...)` defaults to `CPUExecutionProvider`. Pass `providers=(...)` when
you want to opt into other ONNX Runtime execution providers, and keep in mind
that shared runtime stages may still need explicit serialization.

The main expectations are:

- keep the default CPU-only provider list unless you intentionally want a
  different ONNX Runtime execution provider setup
- set `providers=(...)` in the order you want ONNX Runtime to try them when
  you override the default
- use `serialize=True` when one shared `Infer` stage can be reached
  concurrently from multiple workers
- expect `Distribute()` to copy per-sample arrays so downstream mutation does
  not alias the original batched output

## Further Reading

- [`INDEX.md`](./INDEX.md) for the full surface catalog
- [`docs/README.md`](../../../docs/README.md) for the shared framework docs index
- [`examples/README.md`](../../../examples/README.md) for runnable pipeline entry points
  - [`examples/run_yolo8_onnx.py`](../../../examples/run_yolo8_onnx.py)
  - [`examples/run_yolo8_batch.py`](../../../examples/run_yolo8_batch.py)
