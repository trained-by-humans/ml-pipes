# Performance Guide

This document covers the three levers for scaling inference throughput: concurrency,
batching, and serialization. Each section explains the concept, how it affects
performance, and when to use it.

### Key metrics

To optimize performance, you need to understand the three key metrics:

**Throughput** — how many images the system processes per second across all threads.
This is what you optimize when running a service under load.

**Latency** — how long a single request takes end to end. 

**Hardware utilization** — whether the compute units (CPU cores, GPU stream processors)
are idle between inference calls. The goal is to keep inference running continuously
with no gaps.

> [!TIP]
Maximizing performance is often a trade-off; For example batching improves throughput but increases latency because a request must wait for other requests to join the batch:
> Low latency favors small or no batches and few workers; 
> High throughput favors concurrency and (on GPU) larger batches.

Now that you understand the base operation and key metrics, let's see how we can optimize them:

### Baseline: sequential processing

Without any of the techniques below, requests are processed one at a time. Each stage
must finish before the next begins, and the next request cannot start until the current
one is fully complete:

```
Request 1  [preprocess]──[infer]──[postprocess]
Request 2                                       [preprocess]──[infer]──[postprocess]
Request 3                                                                            [preprocess]──[infer]──[postprocess]
           ──────────────────────────────────────────────────────────────────────────────────────────────────────────────▶ time
```

### Concurrency

Multiple requests flow through the pipeline at the same time, each on its own thread:

```
Thread 1  ──[preprocess]──[infer]──[postprocess]──▶
Thread 2     ──[preprocess]──[infer]──[postprocess]──▶
Thread 3        ──[preprocess]──[infer]──[postprocess]──▶
```

**Performance impact**

Concurrency is the cornerstone of every performance improvement in this guide. Batching
only helps if multiple images arrive at the inference boundary simultaneously — that
requires concurrent preprocessing. Serialization only has meaning when there are
multiple concurrent inference calls to manage. Without concurrency, each technique
either has no effect or reverts to the sequential baseline.

With concurrent requests, the hardware is kept continuously fed — while one request is
being inferred, others are being preprocessed, so there are no idle gaps. On GPU, this
overlap is especially effective: the accelerator runs inference while the CPU handles
the next batch's preprocessing entirely in parallel.

**When to use**

Always — for both CPU and GPU deployments. The right number of workers is directly tied
to batch size; see the Batching section for the sizing guidance.

**In the pipeline**

Wrap any pipeline in a thread pool. The pipeline is stateless across calls, so
concurrent execution requires no changes to the pipeline itself.

```python
from concurrent.futures import ThreadPoolExecutor

pipeline = Pipeline([Decode(), Resize(input_size), Normalize(), Infer(model), ...])

with ThreadPoolExecutor(max_workers=8) as pool:
    results = list(pool.map(pipeline, image_paths))
```

**Caveat**

**Memory footprint.** Each worker thread holds its own state and handle its own share of data: the
decoded image, resized tensor, normalized tensor, and any context values stored along
the way. Running N workers means N instances of that data live in memory simultaneously.
Size your worker count with available memory in mind, not just CPU headroom.

**Shared state in operators.** The pipeline guarantees safe concurrency only if each
operator's `__call__` is stateless. A naive implementation that reuses a buffer across calls without thread safety will silently corrupt results
under concurrent load with no error — each thread overwrites the other's data.

### Batching

Multiple inputs are stacked into a single tensor and sent through the model in one call.

```
[preprocess] ─┐
[preprocess] ─┼─ [data1 ,data2, ...] ─ [Collate] ─ [Infer  N=4] ─ [Distribute] ─┬─ [postprocess] ──▶
[preprocess] ─┤                                                                 ├─ [postprocess] ──▶
[preprocess] ─┘                                                                 ├─ [postprocess] ──▶
                                                                                └─ [postprocess] ──▶
```

**Performance impact**

On **GPU**, batching is the primary throughput lever. Tensor cores are most efficient
when given large operands, and memory bandwidth is amortized across more samples.
The larger the batch, the better the hardware utilization — up to the model's memory
limit.

On **CPU**, batching is typically counter-productive. Inference time tends to grow
super-linearly with batch size due to cache pressure and memory bandwidth limits.
Concurrency is the more effective lever on CPU.

**When to use**

| Device | Recommendation                            |
|--------|-------------------------------------------|
| GPU    | Use batching. Start small and profile up. |
| CPU    | Prefer concurrency over batching.         |

Batching adds a hard latency floor: a request must wait until `size` requests have
arrived (or `timeout` fires) before inference starts. Set `timeout` to bound worst-case
wait time for partial batches. 

For batching to be effective, you need enough requests and workers to keep the inference stage
continuously fed. While one batch is being inferred, the next batch must already be
preprocessing. The minimum to guarantee this is:

```
workers < batch_size -> Batches will be never full, the batch always timeouts.
workers == batch_size -> Work become almost sequential, the gain is limited.
workers >= batch_size × 2 -> Not only preprocess is parallel, but always the batches will be ready in parallel.
```

With fewer workers, inference finishes before the next batch is ready and the hardware
idles — negating the benefit of batching entirely.

**In the pipeline**

Add `Batch`/`UnBatch` markers around the inference region. Depends on your operation, add `Collate` to stack individual
tensors into one; `Distribute` splits the output back.

```python
pipeline = Pipeline([
    Decode(),
    Resize(input_size),
    Store("transform", index=1),
    Pick(0),
    Normalize(),
    Batch(size=4, timeout=0.05),    # ◀ threads rendezvous here
    Collate(),                      # list[TensorPayload] → Single batched TensorPayload
    Infer(model),
    Distribute(),                   # Single batched RuntimeOutputs → list[RuntimeOutputs]
    UnBatch(),                      # ◀ each thread resumes with its own result
    Recall("transform"),
    ProjectBoxes(),
    ToDetections(),
])

with ThreadPoolExecutor(max_workers=8) as pool:
    results = list(pool.map(pipeline, image_paths))
```

### Serialization

A lock around the inference call ensures only one execution runs at a time.

```
Thread 1  ──[preprocess]────|infer|────[postprocess]──▶
Thread 2    ──[preprocess]────────|infer|────[postprocess]──▶
Thread 3       ──[preprocess]───────────|infer|────[postprocess]──▶
```

**Performance impact**

Serialization is a concurrency management tool — it only exists because multiple threads
are running simultaneously. With `serialize=True`, each call gets exclusive access 
to the hardware, but the hardware sits idle between calls while threads hand off
the lock. With `serialize=False`, concurrent calls share resources, keeping the
hardware busy at the cost of some contention.

In most cases `serialize=False` (the default) wins on throughput because the hardware
utilization gain outweighs the contention cost. But if profiling shows that concurrent
calls on your hardware degrade individual call performance enough to offset that gain,
`serialize=True` gives back predictable, contention-free execution.

**When to use**

`serialize=False` is the default. Enable `serialize=True` only if profiling shows that
concurrent inference calls are actively hurting throughput on your specific hardware.

**In the pipeline**

Pass `serialize=True` to `Infer` to opt in to the lock.

```python
...
Normalize(),
Infer(model_path, serialize=True),
Extract("output0", as_="preds"),
...
```

### How it all works internally

**Concurrency is free** because `Pipeline.__call__` creates a fresh `Context` on every
invocation. There is no shared mutable state between calls — threads never touch each
other's data.

**Batching is built into the pipeline execution loop.** `Pipeline._scan_batch_pairs()`
maps each `Batch` index to its matching `UnBatch` index at construction time. At
runtime, the first thread to fill the batch becomes the *leader* and runs the batch
region (everything between `Batch` and `UnBatch`). All other threads (*waiters*) block
on a `threading.Event` and skip the region entirely. The leader distributes results
through that event before continuing. Two concurrent batches can run simultaneously
because each leader's state is isolated in `threading.local`.

**Serialization is a context manager swap.** `Infer` holds either a `threading.Lock`
(when `serialize=True`) or a `contextlib.nullcontext` (when `serialize=False`). The
call site is identical in both cases — only the lock object differs.
