# ml-pipes-core Index

For framework-wide operator concepts, see
[`docs/OPERATORS.md`](../../../docs/OPERATORS.md). For the cross-package
package catalogs, see
[`docs/OPERATORS.md#package-catalogs`](../../../docs/OPERATORS.md#package-catalogs).

## User-Facing Modules

| Module | Notes |
|---|---|
| `ml_pipes.core` | Owns `Pipeline`, `Operator`, and pipeline-composition helpers such as `embed()` and `inline()`. |
| `ml_pipes.standard` | Owns the generic reusable operators listed below. |

## Core Pipeline Primitives

| API | Notes |
|---|---|
| `Pipeline([...])` | Ordered execution harness around operator boundaries. |
| `Operator` | Decorator for self-describing callable classes. |
| `embed(p)` / `Embed(p)` | Joins a child pipeline as one isolated step. |
| `inline(p)` / `Inline(p)` | Flattens a child pipeline into the parent at construction time. |

## Selection And Context

| Operator | Input -> Output | Notes |
|---|---|---|
| `Select(*selector)` | `T` -> selected value | Projects from tuples, mappings, objects, or attribute paths. |
| `Pick(*indices)` | tuple -> selected element(s) | Tuple shorthand for routing one or more tuple outputs forward. |
| `Store(name, source=...)` | `T` -> `T` | Saves the current value or a selected subvalue into context. |
| `Recall(name, prepend=False)` | `T` -> tuple with stored value appended or prepended | Re-inserts a stored value into the flowing boundary. |

## Regions And Parallelism

| Operator | Input -> Output | Notes |
|---|---|---|
| `Batch(size, timeout)` | region opener | Groups per-item calls into bounded batches. Pair with `UnBatch()`. |
| `UnBatch()` | region closer | Closes a `Batch` region and returns to the per-item boundary. |
| `Scatter(max_concurrency)` | `list[T]` -> region over `T` | Fans list items out to worker threads. Pair with `Gather()`. |
| `Gather()` | region closer | Collects `Scatter` region outputs back into `list[U]`. |
| `PerItem()` | `Iterable[T]` -> region over `T` | Eagerly runs the enclosed region once per item. Pair with `CollectItems()`. |
| `CollectItems()` | region closer | Materializes `PerItem` outputs as `list[U]`. |
| `LazyPerItem()` | `Iterable[T]` -> region over `T` | Lazily runs the enclosed region once per item. Pair with `StreamItems()`. |
| `StreamItems()` | region closer | Exposes `LazyPerItem` outputs as `Iterable[U]`. |

## Data Preparation

| Operator | Input -> Output | Notes |
|---|---|---|
| `WrapMappingInObject(target, state_factory)` | mapping -> object | Wraps a mapping input into a typed state object. |
| `Map(fn)` | `T` -> `U` | Applies a unary transform to the current value. |
| `MapNotNull(fn)` | `T` -> `U` or drop | Drops items whose mapped result is `None`. |
| `MapValue(fn, source, target)` | `T` -> `T` | Reads a source field, maps it, and stores the result on the current object. |
| `Filter(predicate, source=None)` | `T` -> `T` or drop | Keeps the current value when a predicate matches. |
| `FilterNotNull(source)` | `T` -> `T` or drop | Keeps the current value only when the selected source exists and is non-null. |
| `DropNull()` | `T | None` -> `T` or drop | Drops null current values explicitly. |
| `DistinctBy(fn)` | `Iterable[T]` -> `list[T]` | Deduplicates by a computed key. |
| `Distinct(source)` | `Iterable[T]` -> `list[T]` | Deduplicates by a selected field. |
| `Take(count)` | `Iterable[T]` -> `list[T]` | Materializes the first `count` items. |
| `TakeWhile(predicate)` | `Iterable[T]` -> `list[T]` | Materializes items while a predicate remains true. |

For a concrete data-preparation pipeline built from these operators, see
[`examples/run_sms_spam_prepare.py`](../../../examples/run_sms_spam_prepare.py).
