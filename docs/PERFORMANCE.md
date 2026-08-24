# Performance Guide

This guide is about performance tuning: the execution choices that change
throughput, latency, tail latency, and hardware utilization.

Performance work in `ml-pipes` usually has three connected parts:

- Use [TRACING.md](TRACING.md) when you want to monitor live pipeline behavior
  or inspect where time is going within a call.
- Use this guide when you are ready to tune the pipeline itself through levers
  such as concurrency, batching, and [regions](REGIONS.md).
- Use [BENCHMARKING.md](BENCHMARKING.md) when you want repeated-run
  measurements or to compare configuration changes.

A practical loop is: establish a baseline with
[`Benchmark`](BENCHMARKING.md), use traces when you need to understand the
current behavior, change one execution lever at a time, then benchmark again
to confirm the effect across repeated runs.

## Key metrics

**Throughput** — how many inputs the system processes per second across all threads.
This is what you optimize when running a service under load.

**Latency** — how long a single request takes end to end. This is what you optimize
when running a realtime or streaming service.

**Tail latency** — the slower end of the latency distribution, usually `p95` or
`p99`. This is often the first metric to get worse when batching, queuing, or
synchronization waits are too aggressive.

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

## Choosing A Lever

Start with the simplest lever that matches how work already arrives:

- Many independent requests already reach the pipeline separately: start with
  coarse-grained concurrency.
- One request contains a `list` of independent items: use
  [`Scatter/Gather`](REGIONS.md).
- The caller already produces batches: use coarse-grained batching.
- Separate threads need to meet at one inference stage: use
  [`Batch/UnBatch`](REGIONS.md).

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
the pipeline. This is a [region-based](REGIONS.md) pattern: only the enclosed
steps run concurrently, while steps before and after remain sequential. Each
worker runs the region independently with its own `Context`; the original thread
resumes with `list[results]` after `Gather`.

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
the expected number of items.

> [!TIP]
> Put only independent per-item work inside the `Scatter` region. Shared setup,
> cross-item coordination, and final aggregation usually belong outside it.

```python
pipeline = Pipeline([
    Tile(slice_wh=(640, 640), overlap_wh=(100, 100)),
    Store("tile_rects", source=1),
    Pick(0),
    Scatter(max_concurrency=4),   # ◀ fans out list[ImagePayload] to 4 workers
    Resize((640, 640)),
    Normalize(),
    Infer(model),
    ...,
    Gather(),                     # ◀ rejoins with list[TensorRegistry]
    Recall("tile_rects"),
    Stitch(),
    NMM(iou_threshold=0.5),
])
```

The wall-clock time of the `Scatter` span equals the slowest worker. The child trace
shows the average worker latency, making it easy to spot imbalance.

> [!CAUTION]
> Some inference runtimes already use internal threading or stream
> scheduling. Combined with pipeline-level concurrency or `Scatter`, this can
> cause oversubscription. If throughput degrades at higher `max_concurrency`,
> tune the runtime's own parallelism settings and the pipeline worker count
> together.

## Batching

Multiple inputs are grouped into one batched value and sent through the model
in one call. Often that batched value is a tensor produced by `Collate`:

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
operator. This is a [region-based](REGIONS.md) pattern: steps before and after
the batch region run independently per thread, and only the enclosed region is
batched. `Batch/UnBatch` does not create concurrency by itself; it coordinates
callers that are already in flight. The first thread to fill the batch becomes
the *leader* and runs the batch region for the whole group. All other threads
wait and resume with their individual result after `UnBatch`.

For this to be effective, enough concurrent requests must be in flight to keep the
inference stage continuously fed:

```
workers < batch_size  -> batches are often partial (timeout-driven)
workers ≈ batch_size  -> batching works but overlap is limited
workers > batch_size  -> better overlap and more consistent full batches
```

`Batch`/`UnBatch` handle synchronization (control flow); `Collate`/`Distribute` handle
data transformation.

> [!TIP]
> Put only work that benefits from a batch inside the `Batch` region. Keep
> independent per-item preprocessing before it when you still want thread-level
> overlap.

```python
pipeline = Pipeline([
    Decode(),
    Resize(input_size),
    Store("transform", source=1),
    Pick(0),
    Normalize(),
    Batch(size=4, timeout=0.05),    # ◀ threads rendezvous here
    Collate(),
    Infer(model),
    Distribute(),
    UnBatch(),                      # ◀ each thread resumes with its own result
    Recall("transform"),
    ProjectBoxes(),
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

That trade-off plays out differently by hardware:

- On **GPU**, the two complement each other: concurrency keeps preprocessing
  busy while the GPU runs the batch; batching amortizes kernel launch overhead
  and keeps tensor cores fed.
- On **CPU**, concurrency alone is usually sufficient. Batching collapses the
  cross-core parallelism that makes concurrency effective and may increase
  memory pressure without a compensating gain.

Serialization is orthogonal to both — it is a contention-management tool that 
only matters once concurrency is already in place.

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
[item1, item2, ..., item16] → Scatter ─┼─  item2  ─┼─ Batch(size=4) → Infer → UnBatch → Gather → results[16]
                                       ├─  ...    ─┤
                                       └─  item16 ─┘
```

**Align incoming batches before inference.** Separate threads may arrive with
small or uneven batches. A `Batch` operator can align those arrivals at one
inference stage so the model still runs on a consistent target batch size:

```
Thread 1: [item1, item2, item3]       ─┐                                      ┌─ [result1, result2, result3]
Thread 2: [item4, item5, ..., item8]  ─┼─ Batch(size=8) → Infer → UnBatch ───┼─ [result4, result5, ..., result8]
Thread 3: [...]                       ─┘                                      └─ [...]
```
