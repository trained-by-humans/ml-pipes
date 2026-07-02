# Documentation

Shared `ml-pipes` reference docs, tutorials, and design notes live here.
Package-owned guides live under `packages/<name>/docs/`.

## Start Here

- [../README.md](../README.md) for the framework overview and quick start
- [PACKAGES.md](PACKAGES.md) to choose the right package and import surface
- [../examples/README.md](../examples/README.md) to run a concrete example first

## Understand The Pipeline Model

- [OPERATORS.md](OPERATORS.md) for operator boundaries, naming, and design rules
- [COMPOSITION.md](COMPOSITION.md) for pipeline composition, value shapes, and
  `embed` / `inline`
- [REGIONS.md](REGIONS.md) for batching, scatter/gather, and custom region
  semantics
- [SCAFFOLDING.md](SCAFFOLDING.md) to wrap a new model inside an explicit
  pipeline

## Tooling

- [VALIDATION.md](VALIDATION.md) for static pipeline contract checking
- [TRACING.md](TRACING.md) for one-run timing and collectors
- [BENCHMARKING.md](BENCHMARKING.md) for repeated-run measurement and sweeps
- [PERFORMANCE.md](PERFORMANCE.md) for practical guidance on batching,
  concurrency, and throughput tradeoffs

## Design And Architecture

- [DESIGN.md](DESIGN.md) for the conceptual model and rationale
- [ARCHITECTURE.md](ARCHITECTURE.md) for ownership boundaries and runtime shape

## Package-Owned Docs

- [../packages/core/docs/INDEX.md](../packages/core/docs/INDEX.md)
- [../packages/tensor/docs/INDEX.md](../packages/tensor/docs/INDEX.md)
- [../packages/vision/docs/INDEX.md](../packages/vision/docs/INDEX.md)
- [../packages/onnx/docs/INDEX.md](../packages/onnx/docs/INDEX.md)
- [../packages/torch/docs/INDEX.md](../packages/torch/docs/INDEX.md)
