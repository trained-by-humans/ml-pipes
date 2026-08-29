# Documentation

Shared `ml-pipes` reference docs, tutorials, and design notes live here.
Package-owned guides live under `packages/<name>/docs/`.

## Start Here

- [../README.md](../README.md) for the framework overview and quick start
- [../examples/README.md](../examples/README.md) to run a concrete example first
- [PACKAGES.md](PACKAGES.md) to choose the right package and import surface

## Understand The Pipeline Model

- [OPERATORS.md](OPERATORS.md) for operator boundaries, naming, and design rules
- [COMPOSITION.md](COMPOSITION.md) for pipeline composition

## Design And Architecture

- [DESIGN.md](DESIGN.md) for the conceptual model and rationale
- [ARCHITECTURE.md](ARCHITECTURE.md) for ownership boundaries and runtime shape

## Tooling

- [VALIDATION.md](VALIDATION.md) for static pipeline contract checking
- [INSPECTION.md](INSPECTION.md) for one-run output capture and rendering
- [TRACING.md](TRACING.md) for one-run timing and collectors
- [BENCHMARKING.md](BENCHMARKING.md) for repeated-run measurement and sweeps

## Advanced Topics

- [REGIONS.md](REGIONS.md) for batching, scatter/gather, and custom region
  semantics
- [PERFORMANCE.md](PERFORMANCE.md) for practical guidance on batching,
  concurrency, and throughput tradeoffs
- [SCAFFOLDING.md](SCAFFOLDING.md) to wrap a new model inside an explicit
  pipeline

## Package-Owned Docs

- [../packages/core/docs/INDEX.md](../packages/core/docs/INDEX.md)
- [../packages/tensor/docs/INDEX.md](../packages/tensor/docs/INDEX.md)
- [../packages/vision/docs/INDEX.md](../packages/vision/docs/INDEX.md)
- [../packages/onnx/docs/INDEX.md](../packages/onnx/docs/INDEX.md)
- [../packages/torch/docs/INDEX.md](../packages/torch/docs/INDEX.md)
- [../packages/supervision/docs/INDEX.md](../packages/supervision/docs/INDEX.md) (proposed)
- [../packages/openai/docs/INDEX.md](../packages/openai/docs/INDEX.md) (proposed)
- [../packages/langchain/docs/INDEX.md](../packages/langchain/docs/INDEX.md) (proposed)
