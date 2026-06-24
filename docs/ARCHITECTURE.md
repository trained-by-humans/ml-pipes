# Architecture

This document explains how `ml-pipes` works internally: how a pipeline is represented, how values move through it at runtime, how batching/scatter regions are executed, and how validation and tracing are layered on top.

## Overview

At the highest level, the project is a small execution engine for composable inference pipelines.

The core idea is:

1. A pipeline is just an ordered list of operators.
2. Data moves through that list as a flowing value plus side-channel context.
3. Each operator transforms the flowing value, interacts with context, or both.

Every pipeline invocation carries two main kinds of runtime data:

- flowing value: the value currently moving through the pipeline
- side-channel context: an immutable key/value store used by context-aware operators

The engine itself is deliberately small:

```text
input + context
  -> operator 0: output
  -> operator 1: output
  -> operator 2: ...
  -> ...
  -> output
```

## Main Building Blocks

### 1. Execution engine

`Pipeline` in `core.py` is the runtime and validation engine. It is responsible for:

- storing the operator list
- flattening `Inline(...)` markers at construction time
- executing operators in order
- stepping into bounded regions
- validating structure, context scope, and type contracts
- attaching traces to a collector

`Embed`, `Inline`, `embed()`, and `inline()` also live here because composition changes how the engine sees the operator list.

### 2. Runtime data model

`types.py` defines the small set of domain payloads:

- `ImagePayload`: image array plus color space and layout metadata
- `TensorPayload`: tensor array plus layout and dtype metadata
- `RuntimeOutputs`: named model outputs exactly as returned by ONNX Runtime
- `TensorRegistry`: mutable named store used during postprocessing
- `ResizeTransform`: enough information to map model-space coordinates back to the source image
- `Detections` and `Segmentations`: final structured outputs

This split is intentional:

- immutable dataclasses represent coarse pipeline stages cleanly
- `TensorRegistry` is the mutable named workspace used for tensor-heavy postprocessing

### 3. Context system

`context.py` provides:

- `Context`: immutable mapping with `store()`/`load()`/`merge()`
- `ContextOp`: operator protocol for steps that need access to the side-channel state
- `Store`: saves part or all of the current value under a key
- `Recall`: injects a stored value back into the current stream

The engine treats `ContextOp` differently from normal callables: it calls `apply(current, context)` and receives both a new value and a new `Context`.

### 4. Region system

Regions are the mechanism that lets a pipeline temporarily switch from plain sequential execution to a bounded sub-execution strategy.

The architectural pieces are:

- `RegionOpener` / `RegionCloser`: markers that define a bounded region in the operator list
- `Batch` / `UnBatch`: coordinate multiple concurrent callers into one batched region execution
- `Scatter` / `Gather`: fan a `list[T]` out to worker threads and join the results back into `list[U]`

The important point is that regions are not a separate pipeline type. They are embedded directly in the same operator list and delegated by the main execution loop.

### 5. Standard operators

The operator library supplies the concrete steps that the engine composes: preprocessing, inference, registry transforms, output conversion, context interaction, region coordination, tiling, and side effects.

The full operator catalog and per-operator semantics belong in
[`OPERATORS.md`](OPERATORS.md). The important architectural point here is that
the engine does not hard-code model families or tasks; it just executes a list
of small operators.

## Execution Flow


`Pipeline.__call__` does three things:

1. Initializes per-call tracing state when tracing is enabled.
2. Calls `_execute(value, trace=trace)`.
3. In a `finally`, delivers the trace to the configured collector if tracing is enabled.

That `finally` matters: a failing operator still produces a trace containing completed spans and the error span.

### `_execute()`

`_execute()` is the core loop. For a given operator range it:

1. creates a fresh `Context`
2. starts from the supplied input value
3. walks operators from `start` to `end`
4. dispatches normal operators through `_step()`
5. dispatches region-opening operators through `_step_into_region()`

The fresh-context behavior is important:

- every top-level pipeline call gets a fresh context, which is why embedded pipelines are isolated when `Embed.__call__` invokes a separate `Pipeline.__call__`
- every region sub-execution also gets a fresh context for that sub-run, which is why region bodies such as scatter workers are naturally isolated

### `_step()`

For each non-region operator:

1. if it is a `ContextOp`, call `apply(current, context)`
2. otherwise, inspect the callable signature and build positional arguments from `current`
3. record a `StepSpan`
4. if the operator raises, record an error span and re-raise

Argument routing is intentionally simple:

- if the next operator takes one positional parameter, the whole `current` value is passed as-is
- if it takes multiple positional parameters, `current` must be a tuple of matching arity

That gives the library its tuple-unpacking composition model without extra wrappers.

### `_step_into_region()`

For `RegionOpener` subclasses, the engine:

1. Finds the matching closing operator.
2. Builds a bounded `execute_region(value, child_trace)` closure that can only run that subrange.
3. Calls `operator.run_region(...)`.
4. Skips the operator index forward to the matching closer.

So regions are not interpreted by the main loop one operator at a time. The
opener takes control and decides how the enclosed slice executes. This is a
runtime detail only; static typing can still treat the region boundaries as
linear `In -> Out` steps in the operator list.

## Composition

The engine supports two composition modes:

- merge: flatten operators into one runtime list with shared `Context`
- join: keep a child pipeline as an isolated step with its own `Context`

Architecturally, the distinction matters because it changes whether the engine sees one flat operator list or a boundary represented by `Embed`.

The full composition semantics and user-facing APIs belong in
[`COMPOSITION.md`](COMPOSITION.md).

## Validation

`Pipeline.validate()` is the static safety layer on top of the runtime engine. Internally it checks:

- region structure
- context scope
- left-to-right type contract compatibility
- optional strict-mode rejection of unresolved `Any`

The architectural point is that validation mirrors actual runtime semantics: region nesting, context isolation, and boundary typing are checked the same way the engine will execute them.

The full validation rules, examples, and strict-mode behavior belong in
[`VALIDATION.md`](VALIDATION.md).

## Regions

Regions are the main abstraction that turns a plain sequential pipeline into a concurrent/batched system without changing the surrounding engine.

### Batch / UnBatch

`Batch` owns a `BatchGate`, which coordinates multiple concurrent callers reaching the same pipeline instance.

Runtime behavior:

1. Each caller entering `Batch` calls `gate.enter(current)`.
2. Threads accumulate until either:
   - `size` samples arrive, or
   - `timeout` expires for the current batch
3. One thread becomes the leader and receives all pending inputs as a list.
4. Followers block waiting for the leader’s result distribution.
5. The leader executes the enclosed region once on the batch.
6. At `UnBatch`, results are distributed back to individual waiting callers.

The typical batched region is:

```text
sample
  -> Batch
  -> list[sample]
  -> Collate
  -> Infer
  -> Distribute
  -> UnBatch
  -> sample_result
```

The engine does not have a special case for `UnBatch`. `Batch.run_region()` executes the bounded slice, then uses `BatchGate.distribute()` to hand each caller its own post-region value.

Key design points:

- batching happens across concurrent invocations of the same pipeline object
- only the leader runs the region body
- followers do not re-execute the region
- exceptions are propagated to all waiting followers through `distribute_exception()`

### Scatter / Gather

`Scatter` owns a `ScatterGate`, backed by a `ThreadPoolExecutor`.

Runtime behavior:

1. The current value must be `list[T]`.
2. `Scatter.run_region()` submits one worker task per item.
3. Each worker runs the enclosed region independently through the bounded `execute_region()` closure.
4. Each worker gets a fresh `Context` because each sub-execution calls `_execute()`.
5. `Gather` waits for all worker entries to complete, preserving submission order.
6. The result after the region is `list[U]`.

Typical pattern:

```text
list[item]
  -> Scatter
  -> item
  -> per-item region
  -> Gather
  -> list[result]
```

Unlike `Batch`, scatter is fan-out within one invocation rather than coordination across multiple invocations.

### Why regions matter

The region abstraction keeps the main engine small:

- `Pipeline` only needs to detect openers and delegate
- concurrency policies live in operator-specific code
- validation still uses one uniform nesting model

Tiling is a good example of this design style: it is built from ordinary composition (`Tile`, `Scatter`, `Gather`, `Stitch`, `NMM`) rather than a special engine mode.

## Tracing

Tracing is opt-in and decoupled from execution. Each invocation builds its own trace, operators append spans as they run, and the finished trace is delivered to a collector after the call completes or fails.

Architecturally, regions can attach child traces, so batch and scatter preserve nested timing structure without changing the main execution loop.

The collector types, output format, and tracing API belong in
[`TRACING.md`](TRACING.md).

## Example Pipelines

The examples are not just demos; they also show the intended assembly patterns:

- [`examples/run_yolo8_onnx.py`](/Users/esbati.keivan/PycharmProjects/InferencePipeline/examples/run_yolo8_onnx.py): standard single-image detection
- [`examples/run_batch_yolo8_onnx.py`](/Users/esbati.keivan/PycharmProjects/InferencePipeline/examples/run_batch_yolo8_onnx.py): cross-invocation batching with `Batch`/`UnBatch`
- [`examples/streaming/run_yolo8_webcam.py`](/Users/esbati.keivan/PycharmProjects/InferencePipeline/examples/streaming/run_yolo8_webcam.py): embedding a reusable inference sub-pipeline into a live stream loop
- [`examples/streaming/run_shibuya_counter.py`](/Users/esbati.keivan/PycharmProjects/InferencePipeline/examples/streaming/run_shibuya_counter.py): threaded streaming, optional tiling, and throughput tracing
- [`examples/run_yolo8_tracing.py`](/Users/esbati.keivan/PycharmProjects/InferencePipeline/examples/run_yolo8_tracing.py): tracing API usage

### Typical Detection Flow

A standard image detection example, such as [`examples/run_yolo8_onnx.py`](/Users/esbati.keivan/PycharmProjects/InferencePipeline/examples/run_yolo8_onnx.py), looks like this:

```text
Path
  --LoadFile--> bytes
  --Decode--> ImagePayload
  --Resize--> (ImagePayload, ResizeTransform)
  --Store("resize_transform", index=1)--> (ImagePayload, ResizeTransform)
  --Pick(0)--> ImagePayload
  --Normalize--> TensorPayload
  --Infer--> RuntimeOutputs
  --Extract--> TensorRegistry
  --tensor ops / score ops / box ops--> TensorRegistry
  --NMS--> TensorRegistry
  --Recall("resize_transform")--> (TensorRegistry, ResizeTransform)
  --ProjectBoxes--> TensorRegistry
  --ToDetections--> Detections
```

Two patterns show up repeatedly across examples:

- use `Store` before a destructive transform, then `Recall` later
- keep model-specific logic localized to the operators between `Infer` and the final output conversion

That is why the project can reuse large parts of the pipeline across YOLO, RF-DETR, Mask R-CNN, and tiled/streaming examples.

## Extending The System

The main ways to extend the system are:

### Add a new plain operator

Implement `__call__` with type annotations.

Use this when the operator is just a transformation of the current value.

### Add a new context-aware operator

Implement `ContextOp`.

Use this when the operator must read or write the side-channel `Context`.

### Add a generic operator with precise validation

Implement `resolve_contract(...)`.

Use this when the real output type depends on upstream types or tuple routing.

### Add a new side-effect operator

Subclass `SideEffectOp`.

Use this when the operator performs an effect but should pass the value through unchanged and still work under strict validation.

### Add a new execution strategy

Implement a new `RegionOpener`/`RegionCloser` pair plus its coordination primitive.

This is the architectural pattern used by both batching and scatter/gather.

## Design Tradeoffs

The project makes a few deliberate tradeoffs:

- It prefers explicit dataflow over hidden object state.
- It uses Python type annotations as lightweight contracts rather than a full static type system.
- It keeps the engine small and pushes specialized behavior into operators.
- It accepts a mutable `TensorRegistry` in postprocessing for ergonomic tensor manipulation, while keeping higher-level payloads immutable.
- It uses composition (`inline`, `embed`, `+`, `>>`) rather than deep model-specific inheritance.

The result is a system that is easy to read from top to bottom: most behavior is visible directly in the operator list, while the engine underneath remains compact.

## Reading Guide

If you want to understand the project quickly, read the code in this order:

1. [`src/ml_pipes/core.py`](/Users/esbati.keivan/PycharmProjects/InferencePipeline/src/ml_pipes/core.py)
2. [`src/ml_pipes/context.py`](/Users/esbati.keivan/PycharmProjects/InferencePipeline/src/ml_pipes/context.py)
3. [`src/ml_pipes/ops.py`](/Users/esbati.keivan/PycharmProjects/InferencePipeline/src/ml_pipes/ops.py)
4. [`src/ml_pipes/batch.py`](/Users/esbati.keivan/PycharmProjects/InferencePipeline/src/ml_pipes/batch.py)
5. [`src/ml_pipes/scatter.py`](/Users/esbati.keivan/PycharmProjects/InferencePipeline/src/ml_pipes/scatter.py)
6. [`src/ml_pipes/tracing.py`](/Users/esbati.keivan/PycharmProjects/InferencePipeline/src/ml_pipes/tracing.py)
7. [`tests/test_core.py`](/Users/esbati.keivan/PycharmProjects/InferencePipeline/tests/test_core.py), [`tests/test_batch.py`](/Users/esbati.keivan/PycharmProjects/InferencePipeline/tests/test_batch.py), [`tests/test_scatter.py`](/Users/esbati.keivan/PycharmProjects/InferencePipeline/tests/test_scatter.py), [`tests/test_tracing.py`](/Users/esbati.keivan/PycharmProjects/InferencePipeline/tests/test_tracing.py)

That path takes you from the engine, to its state model, to the operators built on top of it, to the tests that define the intended behavior.
