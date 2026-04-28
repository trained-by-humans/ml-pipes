# Performance Guide

This document covers the levers for scaling inference throughput.

## Key metrics

**Throughput** — how many inputs the system processes per second across all threads.
This is what you optimize when running a service under load.

**Latency** — how long a single request takes end to end. This is what you optimize
when running a realtime or streaming service.

**Hardware utilization** — whether the compute units (CPU cores, GPU stream processors)
are idle between inference calls. The goal is to keep inference running continuously
with no gaps.

> [!TIP]
> Maximizing performance is often a trade-off:
> - Low latency favors small or no batches and few workers
> - High throughput favors concurrency and (on GPU) larger batches

## Baseline: sequential processing

Without any of the techniques below, requests are processed one at a time:

```
Request 1  [preprocess]──[infer]──[postprocess]
Request 2                                      [preprocess]──[infer]──[postprocess]
Request 3                                                                          [preprocess]──[infer]──[postprocess]
Time       ─────────────────────────────────────────────────────────────────────────────────────────────────────────────▶ 
```

## Concurrency

Multiple units of work run simultaneously on separate threads:

```
Thread 1  ──[preprocess]──[infer]──[postprocess]──▶
Thread 2     ──[preprocess]──[infer]──[postprocess]──▶
Thread 3        ──[preprocess]──[infer]──[postprocess]──▶
```

Concurrency keeps the hardware continuously fed — while one unit is being inferred,
others are being preprocessed, so there are no idle gaps.

**Performance impact**

On **CPU**, concurrency is the primary throughput lever. Each thread runs independently
across cores with no coordination overhead.

On **GPU**, concurrency overlaps CPU preprocessing with GPU inference, keeping both
busy simultaneously.

**Caveats**

**Memory footprint.** Running N concurrent units means N instances of in-flight data
live in memory simultaneously. Size your worker count with available memory in mind,
not just CPU headroom.

**Shared state in operators.** The pipeline guarantees safe concurrency only if each
operator's `__call__` is stateless. A naive implementation that reuses a buffer across
calls without thread safety will silently corrupt results under concurrent load.

### Coarse-grained concurrency

The caller runs multiple requests through the same pipeline simultaneously using a
thread pool. Concurrency applies to the entire pipeline as a unit — every step from
decode to postprocess runs concurrently across requests. The pipeline is stateless
across calls — no changes needed.

```python
from concurrent.futures import ThreadPoolExecutor

pipeline = Pipeline([Decode(), Resize(input_size), Normalize(), Infer(model), ...])

with ThreadPoolExecutor(max_workers=8) as pool:
    results = list(pool.map(pipeline, image_paths))
```

### Fine-grained concurrency (Scatter/Gather)

A single request fans out a `list[T]` to N worker threads at a specific point inside
the pipeline. Only the enclosed region runs concurrently — steps before and after
remain sequential. Each worker runs the region independently with its own `Context`;
the original thread resumes with `list[results]` after `Gather`.

```
                    ┌─ [worker 0: region ops] ─┐
[produce list[T]] ──┼─ [worker 1: region ops] ─┼── [list[U]] ──▶
                    └─ [worker 2: region ops] ─┘
```

This is useful whenever a single request contains independent sub-tasks — for example:
tiled inference (each tile runs through the full inference region concurrently),
multi-task processing (a list of independent jobs), or multi-model fan-out (same input
sent to N models simultaneously).

`max_concurrency` caps the thread pool size. Set it based on available CPU cores and
the expected number of items. The thread pool is persistent across invocations.

```python
pipeline = Pipeline([
    Tile(slice_wh=(640, 640), overlap_wh=(100, 100)),
    Store("tile_rects", index=1),
    Pick(0),
    Scatter(max_concurrency=4),   # ◀ fans out list[ImagePayload] to 4 workers
    Resize((640, 640)),
    Normalize(),
    Infer(model),
    ...,
    ToDetections(),
    Gather(),                     # ◀ rejoins with list[Detections]
    Recall("tile_rects"),
    Stitch(),
    NMM(iou_threshold=0.5),
])
```

The wall-clock time of the `Scatter` span equals the slowest worker. The child trace
shows the average worker latency, making it easy to spot imbalance.

**Caveat:** ONNX Runtime uses multiple threads per inference call internally. When
combined with `Scatter`, this can cause contention — if throughput degrades at higher
`max_concurrency`, reduce either the ONNX inter-op thread count or `max_concurrency`.

## Batching

Multiple inputs are stacked into a single tensor and sent through the model in one call:

```
[preprocess] ─┐                                                                ┌─ [postprocess] ──▶
[preprocess] ─┼─ [data1, data2, ...] ─ [Collate] ─ [Infer N=4] ─ [Distribute] ─┼─ [postprocess] ──▶
[preprocess] ─┤                                                                ├─ [postprocess] ──▶
[preprocess] ─┘                                                                └─ [postprocess] ──▶
```

**Performance impact**

On **GPU**, batching is the primary throughput lever. Larger batches generally improve
hardware utilization up to memory limits.

On **CPU**, batching often provides limited or negative gains. Independent concurrent
inferences typically achieve better throughput by preserving parallelism across cores.

**Caveats**

**Latency floor.** A request must wait until `size` requests have arrived (or `timeout`
fires). Set `timeout` to bound worst-case wait time for partial batches.

**Latency variability.** Under low load, requests may wait for batch formation, increasing
tail latency. The hard floor affects every request even when the pipeline is lightly loaded.

**Reduced concurrency inside the batch region.** Steps inside the batch region run on the
leader thread, not concurrently across workers. For example, if decoding is included in the
batch region, all images are decoded sequentially by the leader instead of in parallel across
threads, negating the benefit of concurrency for that step.

### Coarse-grained batching

The caller groups inputs before passing them to the pipeline. Batching applies to the
entire pipeline — the pipeline receives a pre-formed batch and processes it in one call
with no internal synchronization needed.

```python
pipeline = Pipeline([Collate(), Infer(model), Distribute(), ...])

batch = [image1, image2, image3, image4]
results = pipeline(batch)
```

### Fine-grained batching (Batch/UnBatch)

Concurrent requests rendezvous at a specific point *inside* the pipeline at the `Batch`
operator. Steps before and after the batch region run independently per thread — only the
enclosed region is batched. The first thread to fill the batch becomes the *leader* and
runs the batch region for the whole group. All other threads wait and resume with their
individual result after `UnBatch`.

For this to be effective, enough concurrent requests must be in flight to keep the
inference stage continuously fed:

```
workers < batch_size  -> batches are often partial (timeout-driven)
workers ≈ batch_size  -> batching works but overlap is limited
workers > batch_size  -> better overlap and more consistent full batches
```

`Batch`/`UnBatch` handle synchronization (control flow); `Collate`/`Distribute` handle
data transformation.

```python
pipeline = Pipeline([
    Decode(),
    Resize(input_size),
    Store("transform", index=1),
    Pick(0),
    Normalize(),
    Batch(size=4, timeout=0.05),    # ◀ threads rendezvous here
    Collate(),
    Infer(model),
    Distribute(),
    UnBatch(),                      # ◀ each thread resumes with its own result
    Recall("transform"),
    ProjectBoxes(),
    ToDetections(),
])

with ThreadPoolExecutor(max_workers=8) as pool:
    results = list(pool.map(pipeline, image_paths))
```

## Serialization

A lock around the inference call ensures only one execution runs at a time.

```
Thread 1  ──[preprocess]────|infer|────[postprocess]──▶
Thread 2    ──[preprocess]────────|infer|────[postprocess]──▶
Thread 3       ──[preprocess]───────────|infer|────[postprocess]──▶
```

With `serialize=True`, each call gets exclusive access to the hardware, but the hardware
sits idle between calls while threads hand off the lock. With `serialize=False`
(the default), concurrent calls share resources, keeping the hardware busy at the cost
of some contention.

In most cases `serialize=False` wins on throughput. Enable `serialize=True` only if
profiling shows that concurrent inference calls are actively hurting throughput on your
specific hardware.

```python
Infer(model_path, serialize=True)
```

Serialization is always fine-grained — it targets a single operator, leaving the rest
of the pipeline unaffected.

## Comparing Techniques

| Lever         | Effect on throughput                               | Effect on latency                             | Best for                                                       |
|---------------|----------------------------------------------------|-----------------------------------------------|----------------------------------------------------------------|
| Concurrency   | High — keeps hardware continuously fed             | Neutral to slight increase                    | CPU and GPU; always the starting point                         |
| Batching      | High on GPU; neutral or negative on CPU            | Increases (requests wait for batch formation) | GPU throughput; amortizes kernel launch overhead               |
| Serialization | Neutral to negative — hardware idles between calls | More predictable                              | Rare: when concurrent calls contend badly on specific hardware |

Concurrency and batching pull in opposite directions:

- **Concurrency** increases parallelism — more requests in flight at once, each independent.
- **Batching** reduces parallelism but increases work per call — N requests share one inference call.

On **GPU**, the two complement each other: concurrency keeps preprocessing busy while the GPU runs the batch; batching amortizes kernel launch overhead and keeps tensor cores fed.

On **CPU**, concurrency alone is usually sufficient. Batching collapses the cross-core parallelism that makes concurrency effective and may increase memory pressure without a compensating gain.

Serialization is orthogonal to both — it is a contention-management tool that only matters once concurrency is already in place.

## Combining Techniques

Fine-grained concurrency and batching can be composed freely — `Scatter` and `Batch` can
be nested and combined within a single pipeline. A few useful patterns:

**Parallelize decoding, batch for inference.** A pre-formed batch of encoded images is
scattered so each item is decoded concurrently. After `Gather` the results are already a
list — collating them gives a full batch ready for a single inference call:

```
list[encoded] → Scatter → Decode → Normalize → Gather → Collate → Infer → Distribute
```

**Tile and batch concurrently.** A single image is tiled, then each tile is decoded and
normalized concurrently inside a `Scatter` region. A `Batch` operator inside the region
lets multiple worker threads rendezvous for batched inference before continuing
independently:

```
Image → Tile → Scatter → Decode → Normalize → Batch → Infer → UnBatch → Gather → Stitch
```

**Split a large batch into smaller ones.** An oversized input batch is scattered into
individual items and re-batched at a smaller size — useful when the caller produces batches
larger than the model's optimal batch size:

```
                                       ┌─  item1  ─┐
[item1, item2, ..., item16] → Scatter ─┼─  item2  ─┼─ Batch(size=4) → Infer → UnBatch → Gather -> results[16]
                                       ├─  ...    ─┤
                                       └─  item16 ─┘
```

**Merge small non-uniform batches into larger uniform ones.** Independent small batches from concurrent threads
converge at a `Batch` operator, forming an optimal larger batch:

```
Thread 1: [item1, item2, item3]       ─┐                                      ┌─ [result1, result2, result3]     
Thread 2: [item4, item5, ..., item8 ] ─┼─ Batch(size=8) → Infer → UnBatch →  ─┼─ [result4, result5, ..., result8 ]
Thread 3: [...]                       ─┘                                      └─ [...]                      
```

## How it all works internally

**Coarse-grained concurrency is available by design** because `Pipeline.__call__` creates
a fresh `Context` on every invocation. There is no shared mutable state between calls.

**Fine-grained concurrency** uses a scatter primitive (`ScatterGate`). The original thread
submits one task per item to a `ThreadPoolExecutor`, then blocks until all workers have
deposited their results. Workers run the region operators via the same `_execute` path
as the top-level pipeline, with an isolated `Context` per worker.

**Coarse-grained batching** requires no pipeline changes beyond adding `Collate` and
`Distribute`. Most operators work on the batch dimension transparently — `Resize`,
`Normalize`, `Infer`, and the tensor operators all accept N-dimensional arrays, so a
collated batch flows through the same operators as a single input without modification.

**Fine-grained batching** uses a gate primitive (`BatchGate`). The first thread to fill the
batch becomes the leader and runs the batch region. All other threads block and skip
the region. The leader distributes results via the gate before continuing.

**Serialization** is a context manager swap inside `Infer`. The call site is identical
in both modes — only the behavior of the lock differs.

## Additional note on runtimes

Some inference runtimes use internal parallelism (e.g., multiple threads per inference
call). When combined with pipeline-level concurrency or `Scatter`, this can lead to
resource contention or diminishing returns at higher worker counts.

If you observe scaling limits, consider that performance is influenced not only by the
pipeline but also by how the underlying runtime schedules work internally.
