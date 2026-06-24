# Regions

Regions are bounded sections of a pipeline that temporarily switch execution
from plain step-by-step flow to a specialized strategy such as batching or
fan-out parallelism.

They are still part of the same pipeline. `ml-pipes` does not introduce a
separate pipeline type for batched or parallel execution. Instead, the main
engine delegates the enclosed slice to a region opener when it reaches one.

## What A Region Is

Architecturally, a region is defined by:

- a `RegionOpener`
- a matching `RegionCloser`
- a bounded slice of operators between them

The built-in region pairs are:

- `Batch ... UnBatch`
- `Scatter ... Gather`

At runtime, the main execution loop does not interpret the enclosed operators
one step at a time. It finds the matching closer, builds a bounded
`execute_region(...)` closure for that slice, and calls
`operator.run_region(...)`.

That gives regions two useful properties:

- the outer pipeline still reads as one linear operator list
- each region can own its own execution policy without complicating the main
  engine

From outside the region, the boundary is still treated as a normal `In -> Out`
step in the pipeline.

## Region Isolation

Each region body executes as its own bounded sub-run.

That matters for context scope:

- values stored inside a region stay inside that region
- outer stored keys are not automatically shared into isolated sub-runs
- validation uses the same boundaries when checking `Store` and `Recall`

This is why region structure and region-local context scope can be checked
statically before the pipeline runs.

## Batch / UnBatch

`Batch` owns a `BatchGate`, which coordinates multiple concurrent callers
reaching the same pipeline instance.

Runtime behavior:

1. Each caller entering `Batch` calls `gate.enter(current)`.
2. Threads accumulate until either:
   - `size` samples arrive, or
   - `timeout` expires for the current batch.
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

The engine does not have a special case for `UnBatch`. `Batch.run_region()`
executes the bounded slice, then uses `BatchGate.distribute()` to hand each
caller its own post-region value.

Key design points:

- batching happens across concurrent invocations of the same pipeline object
- only the leader runs the region body
- followers do not re-execute the region
- exceptions are propagated to all waiting followers through
  `distribute_exception()`

## Scatter / Gather

`Scatter` owns a `ScatterGate`, backed by a `ThreadPoolExecutor`.

Runtime behavior:

1. The current value must be `list[T]`.
2. `Scatter.run_region()` submits one worker task per item.
3. Each worker runs the enclosed region independently through the bounded
   `execute_region()` closure.
4. Each worker gets a fresh `Context` because each sub-execution calls
   `_execute()`.
5. `Gather` waits for all worker entries to complete, preserving submission
   order.
6. The result after the region is `list[U]`.

The typical scatter region is:

```text
list[item]
  -> Scatter
  -> item
  -> per-item region
  -> Gather
  -> list[result]
```

Unlike `Batch`, scatter is fan-out within one invocation rather than
coordination across multiple invocations.

## Why Regions Matter

The region abstraction keeps the engine small:

- `Pipeline` only needs to detect openers and delegate
- concurrency policies live in operator-specific code
- validation still uses one uniform nesting model

Tiling is a good example of this design style: it is built from ordinary
composition (`Tile`, `Scatter`, `Gather`, `Stitch`, `NMM`) rather than a
special engine mode.
