# Regions

Regions are bounded sections of a pipeline that temporarily switch execution
from plain step-by-step flow to a specialized strategy such as batching or
fan-out parallelism.

They are still part of the same pipeline. `ml-pipes` does not introduce a
separate pipeline type for batched or parallel execution. Instead, the main
engine delegates the enclosed slice to a region opener when it reaches one.

## What Regions Are

A region is a bounded section of one pipeline. Architecturally, a region has
three parts:

- a `RegionOpener` marking the start of the region
- a matching `RegionCloser`
- a body: the operators between the opener and closer

Minimal example:

```python
from ml_pipes import Batch, Collate, Distribute, Infer, Pipeline, UnBatch


pipeline = Pipeline([
    Decode(),
    Batch(size=8, timeout=0.05),
    Collate(),
    Infer("model.onnx"),
    Distribute(),
    UnBatch(),
    ToDetections(),
])
```

Here, the region is:

- opener: `Batch(...)`
- body: `Collate()`, `Infer(...)`, `Distribute()`
- closer: `UnBatch()`

From outside, a region still behaves like one linear boundary in the pipeline.
The opener describes the boundary before the region and the first value seen by
the region body. The closer describes the boundary at the end of the body and
the value after leaving the region.

## How Regions Work

At runtime, the main execution loop does not interpret the region body one
operator at a time. It:

1. reaches a `RegionOpener`
2. finds the matching `RegionCloser`
3. builds a bounded `execute_region(...)` closure for the enclosed slice
4. calls `opener.run_region(...)`
5. resumes after the closer with the value returned by the opener

Each region body runs as its own bounded sub-execution. That is why context
scope, child traces, and validation stay aligned with the region boundary.

## Why Regions Exist

Regions exist because sometimes "call the next operator" is not the right
execution model.

The design pressure comes from both sides:

- operators should stay focused on their local transform instead of owning
  coordination or execution policy
- `Pipeline` should stay generic instead of hard-coding special logic for
  batching, scatter, streaming, or other reusable execution patterns

Regions sit in the middle. They let the framework support recurring execution
quirks without making ordinary operators or the main engine more complex than
they need to be.

The important design point is that this does not require another pipeline
abstraction. Regions let the framework keep one operator-list model while still
supporting specialized execution.

That gives the system a few useful properties:

- pipelines still read top-to-bottom as one operator list
- execution policy stays local to the region opener instead of spreading into
  the main engine
- validation can reason about the same boundaries the runtime will execute
- tracing can attach child traces to the region span
- higher-level patterns such as tiling can be built by composition instead of
  special engine modes

## Examples

### Batch / UnBatch

`Batch` / `UnBatch` is the cross-invocation case. Multiple callers reach the
same pipeline instance one sample at a time, but the region body runs once on a
list of samples.

Example:

```python
from ml_pipes import Batch, Collate, Distribute, Extract, Infer, NMS
from ml_pipes import Normalize, Pipeline, Resize, ToDetections, UnBatch


pipeline = Pipeline([
    Resize((640, 640)),
    Normalize(),
    Batch(size=8, timeout=0.05),
    Collate(),
    Infer("model.onnx"),
    Distribute(),
    UnBatch(),
    Extract("boxes", "scores", "classes"),
    NMS(),
    ToDetections(),
])
```

What happens at runtime:

- each caller entering `Batch` waits at the same `BatchGate`
- once the gate fills or times out, one leader thread receives the whole batch
- only that leader executes the region body
- `UnBatch` routes one post-region result back to each waiting caller

This pattern is useful for cases such as GPU inference, where the pipeline can
benefit from executing the region body on formed batches instead of one item at
a time.

### Scatter / Gather

`Scatter` / `Gather` is the fan-out case. One pipeline invocation reaches the
region with `list[T]`, and the region body runs once per item.

Example:

```python
from ml_pipes import Extract, Gather, Infer, NMS, Normalize, Pick
from ml_pipes import Pipeline, Recall, Resize, Scatter, Stitch, Store, Tile
from ml_pipes import ToDetections


pipeline = Pipeline([
    Tile(slice_wh=(640, 640), overlap_wh=(100, 100)),
    Store("tile_rects", index=1),
    Pick(0),
    Scatter(max_concurrency=4),
    Resize((640, 640)),
    Normalize(),
    Infer("model.onnx"),
    Extract("boxes", "scores", "classes"),
    NMS(),
    ToDetections(),
    Gather(),
    Recall("tile_rects"),
    Stitch(),
])
```

What happens at runtime:

- `Scatter` submits one worker task per item
- each worker runs the same enclosed operator slice independently
- each sub-execution gets a fresh `Context`
- `Gather` waits for all workers and resumes with `list[U]` in submission order

This pattern is useful when one input naturally expands into many independent
pieces of work, such as tiles, frames, or per-item transforms.

## How To Define A Region

A custom region is another opener / closer pair in the same pipeline model.

### 1. Define the closer

Subclass `RegionCloser[BodyOutputT, OutputT]`.

The closer is usually a stateless marker. Its job is to mark where the region
ends and, when necessary, describe the boundary after the region.

If the closer needs to make the post-region boundary explicit for validation,
implement `resolve_contract(...)`. Simple closers may not need that once the
region boundary can be inferred directly from their generic parameters.

### 2. Define the opener

Subclass `RegionOpener[InputT, BodyInputT]` and set `closing_type` to the
matching closer class.

The opener owns the execution strategy and any coordination state. For example,
`Batch` owns a `BatchGate`, and `Scatter` owns a `ScatterGate`.

Decorate the opener with `@Operator` if you want it to show up in
`repr()` and `describe()` like other configured operators.

### 3. Implement `run_region(...)`

`run_region(...)` is where the region decides how the enclosed body runs.

It receives:

- `current`: the value just before the region
- `label`: the operator label used in traces
- `execute_region`: a bounded callable for the enclosed operator slice
- `trace`: the parent trace object
- `cfg`: tracing configuration

The opener can call `execute_region(...)` once, many times, on different
threads, or after some coordination step. That is the core extension point.

### 4. Return the post-region value

`run_region(...)` returns the value that should appear after the closer. The
engine then resumes at the operator after that closer.

That means the opener controls both:

- how the body executes
- what value the surrounding pipeline sees after the region

### 5. Teach validation the real boundary

Use the opener and closer generics to describe the local boundary first.

- `RegionOpener[InputT, BodyInputT]` means:
  `InputT` before the region, `BodyInputT` for the first operator inside it
- `RegionCloser[BodyOutputT, OutputT]` means:
  `BodyOutputT` for the last operator inside it, `OutputT` after the region

If that is still not precise enough, implement `resolve_contract(...)` on the
opener, the closer, or both. This is how built-in regions express boundaries
such as:

- `Batch`: `T -> list[T]`
- `UnBatch`: `list[T] -> T`
- `Scatter`: `list[T] -> T`
- `Gather`: `T -> list[T]`

Minimal skeleton:

```python
from typing import Any

from ml_pipes import Operator, RegionCloser, RegionOpener


class MyRegionEnd(RegionCloser[str, list[str]]):
    def resolve_contract(
        self,
        current_output: Any | None,
        stored_annotations: dict[str, Any],
        expand_output_annotation: Any,
        validation_error_type: type[Exception],
    ) -> tuple[Any, Any]:
        out = list[current_output] if current_output is not None else list[Any]
        return (Any,), out


@Operator
class MyRegionStart(RegionOpener[list[str], str]):
    closing_type = MyRegionEnd

    def run_region(self, current, label, execute_region, trace, cfg):
        results = []
        for value in current:
            child_trace = ...
            result, child_trace = execute_region(value, child_trace)
            results.append(result)
        return results
```

> [!TIP]
> Read `Batch`, `Scatter`, and `PerItem` for three concrete region styles:
> cross-invocation coordination, fan-out parallelism, and repeated per-item
> execution.
