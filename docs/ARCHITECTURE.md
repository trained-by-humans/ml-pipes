# Architecture

Following up on [DESIGN.md](DESIGN.md), this document is a map of how
`ml-pipes` is split internally. It focuses on ownership boundaries: what each
layer owns, how one invocation moves through the system, and where a change
should usually land.

It is not a guide to using `ml-pipes` features. The detailed user-facing
semantics live in the linked docs.

## Overview

Architecturally, `ml-pipes` is a small generic execution harness around
explicit operator boundaries.

The architecture stays split on purpose:

- `Pipeline` owns generic execution, composition boundaries, and lifecycle
  hooks.
- Operators own task and domain logic.
- `Context` carries side-channel state that should not stay in the flowing
  value.
- Regions own bounded execution strategies such as batching and fan-out.
- Tooling works by reading or validating the same boundaries rather than
  inventing a separate model.

The architectural goal is to provide useful pipeline features while keeping ordinary
operators as simple as possible. In the normal case, an operator should only
need to define its boundary and its task logic; composition, validation,
inspection, tracing, benchmarking, and other cross-cutting behavior should be
handled by the surrounding system.

That keeps the core engine small:

```text
input value + fresh context
  -> operator boundary
  -> operator boundary
  -> bounded region (optional)
  -> operator boundary
  -> output value
```

The pipeline engine itself is payload-agnostic. Image payloads, tensor
payloads, registries, detections, and data-preparation values are ordinary
operator inputs and outputs, not special pipeline modes.

## Main Components

### Pipeline

The pipeline engine owns:

- the ordered operator list
- top-level invocation lifecycle
- dispatch of normal operators, `ContextOp` steps, and region openers
- composition boundaries between merged and embedded pipelines
- trace delivery after a call completes or fails

It does not own:

- model-specific preprocessing or postprocessing
- domain payload semantics
- use-case-specific execution logic beyond generic region delegation

### Operators

Operators are the unit the engine composes and the unit the tooling sees.

Architecturally, that means:

- one operator should own one meaningful input-to-output boundary
- task logic belongs in operators, not in the pipeline engine
- fixed-length tuples are the multi-value routing mechanism between steps
- annotations, labels, and config descriptions give validation, inspection,
  tracing, and description something concrete to work with

For operator boundaries and creation guidelines, see
[OPERATORS.md](OPERATORS.md).

### Context

`Context` is the side-channel state model. It exists so a pipeline can keep one
clear flowing value while still storing values that need to be recovered later.

The important properties are:

- only `ContextOp` steps read or write the `Context`
- each top-level pipeline call starts with a fresh `Context`
- embedded pipelines and region sub-executions are isolated because they run
  through separate bounded execution calls

### Regions

Regions sit between operators and the pipeline engine. They handle execution
strategies that are still generic enough to reuse, but too structural to bury
inside ordinary operators and too use-case-specific to hard-code into
`Pipeline`.

A region:

- is defined by an opener/closer pair inside the same operator list
- owns how the enclosed slice executes
- still presents a normal input/output boundary to the rest of the pipeline

Regions are embedded in the same operator list; they do not create a separate
pipeline type. `Batch`/`UnBatch` and `Scatter`/`Gather` are the main built-in
examples.

For region semantics and examples, see [REGIONS.md](REGIONS.md).

### Composition

Composition is essentially arranging operators into one larger pipeline.
`ml-pipes` supports two ways to do that, and they imply different runtime
boundaries:

- merge (`+`) flattens pipelines into one operator list with one shared
  `Context`
- join (`>>`) preserves a child pipeline boundary through `Embed`, so the child
  runs as an isolated step with its own `Context`

This distinction matters architecturally because it changes both runtime
isolation and what the tooling can see as one step.

For the user-facing composition model, see [COMPOSITION.md](COMPOSITION.md).

## Execution Flow

One pipeline invocation follows the same path whether or not extra tooling is
attached:

1. A top-level call starts with an input value, a fresh `Context`, and
   optionally a trace object.
2. The engine walks the operator list left to right.
3. Normal operators receive the current value. If one step returns a
   fixed-length tuple and the next step expects multiple positional inputs,
   that tuple is routed by position.
4. `ContextOp` steps receive both the current value and the current `Context`,
   and return updated versions of them.
5. When the pipeline reaches a region, it delegates execution of the enclosed
   operators to that region as one bounded unit.
6. If tracing is enabled, spans are recorded for the same operator and region
   boundaries the engine executes.
7. After the call completes or fails, the finished trace is delivered to the
   collector.

That is why validation, inspection, tracing, and benchmarking all line up with
operator boundaries rather than a separate internal graph.

## Tooling

Tooling in `ml-pipes` is layered on top of the same execution and boundary
model. Most tools do not introduce a new executor. They reuse pipeline
execution, operator descriptions, or trace data. Tracing is the exception: it
is built into pipeline execution itself and then supports tools such as
inspection and benchmarking.

### Validation

Validation is the static safety layer over the runtime model. It checks
whether a pipeline is structurally and type-wise coherent before runtime,
without executing the pipeline.

Built on top of:

- the pipeline execution model, so validation checks the same left-to-right
  boundaries the runtime will use
- operator boundaries, especially annotations and `resolve_contract(...)`
- region semantics
- context semantics

Main components:

- `Pipeline.validate()` as the public entry point
- `PipelineValidator`
- typing and signature inspection helpers

For the concrete validation rules and contract-resolution model, see
[VALIDATION.md](VALIDATION.md).

### Tracing

Tracing is the runtime observation layer. It observes runtime behavior without
changing the pipeline structure. It is built into pipeline execution rather
than layered on top of it, and other tools, especially inspection and
benchmarking, build on top of it.

Built into:

- the pipeline execution loop, which emits spans at operator and region
  boundaries
- operator and region boundaries themselves

Main components:

- `InvocationTrace` as the invocation/call trace
- `StepSpan` as step / region trace span
- `TraceCollector`s such as `CaptureCollector`, `PrintCollector`, and
  `AggregateCollector`

For trace lifecycle, built-in collectors, and custom collector patterns, see
[TRACING.md](TRACING.md).

### Inspection

Inspection is built directly on top of tracing. `Pipeline.inspect()` runs the
pipeline once with captured outputs and returns an `InspectionResult`.
`PipelineInspector` then turns that artifact into terminal and HTML/browser
output through formatters and renderers. Core owns the shared inspection
artifact, registry, and renderers, while package-owned types register their
specialized formatters from the owning package at import time.

Built on top of:

- tracing, especially the captured span tree and frozen trace snapshots
- normal pipeline execution

Main components:

- `Pipeline.inspect()` as the public entry point
- `InspectionResult` as the captured inspection artifact
- `PipelineInspector` as the display-oriented inspection layer
- span formatters, output formatters, and renderers

### Benchmarking

Benchmarking is built on top of tracing. It measures repeated-run behavior and
compares variants or configurations without introducing a different runtime.

Built on top of:

- tracing, whose per-run timing data benchmarking aggregates into
  repeated-run measurements
- normal pipeline execution
- factories, which support config-driven sweeps and builder workflows

Main components:

- `Benchmark` as the single-pipeline measurement loop
- `BenchmarkSweep` as the config-sweep execution layer
- `BenchmarkBuilder` as the fluent benchmark setup layer

For benchmarking workflows, sweeps, and CLI usage, see
[BENCHMARKING.md](BENCHMARKING.md). For tuning guidance, see
[PERFORMANCE.md](PERFORMANCE.md).

### Pipeline Description

Pipeline description is a standalone structural tool. It surfaces pipeline
structure without executing the pipeline. `Pipeline.describe()` and
`repr(pipeline)` are the public entry points; both are thin surfaces over the
same structural description model.

Built on top of:

- pipeline objects and composition structure
- operator boundaries, especially operator descriptions

Main components:

- `Pipeline.describe()` / `repr(pipeline)` as the public entry points
- `PipelineDescription` as the structural description artifact
- `OperatorDescription` as the per-operator description model

### Factories

Factories adapt configuration into concrete pipelines or input builders. They
package construction logic without changing how pipeline execution works.

Built on top of:

- pipeline objects and input builders
- config-driven construction of callables

Main components:

- `Factory` as the core concept
- `PipelineFactory` (and its decorator `@pipeline_factory`)
- `DataFactory` (and its decorator `@data_factory`)

### CLI

The CLI is a thin interface layer that exposes `ml-pipes` features as commands.
It does not introduce a separate execution model. It mainly loads references,
resolves factories, parses config, and invokes existing pipeline and
benchmarking surfaces.

Built on top of:

- pipeline execution
- factories, which provide discoverable pipeline and data builders
- standalone tools such as benchmarking

Main components:

- `python -m ml_pipes` as the public entry point
- Commands like `run` and `benchmark`
- module loading, factory discovery, config parsing, and result-saving helpers

## Where To Go Next

This document has mapped how `ml-pipes` is split internally. If you want the
rationale behind these boundaries, return to [DESIGN.md](DESIGN.md).

If your next step is building with the framework, continue to
[SCAFFOLDING.md](SCAFFOLDING.md) or the runnable examples in
[examples/README.md](../examples/README.md).
