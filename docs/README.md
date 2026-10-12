# Documentation

Shared `ml-pipes` reference docs, tutorials, and design notes live here.
Package-owned guides live under `packages/<name>/docs/`.

## Start Here

- [../README.md](../README.md) for the framework overview and quick start
- [../examples/README.md](../examples/README.md) to run a concrete example first
- [packages.md](packages.md) to choose the right package and import surface

## Understand The Pipeline Model

- [operators.md](operators.md) for operator boundaries, naming, and design rules
- [composition.md](composition.md) for pipeline composition

## Design And Architecture

- [design.md](design.md) for the conceptual model and rationale
- [architecture.md](architecture.md) for ownership boundaries and runtime shape

## Tooling

- [validation.md](validation.md) for static pipeline contract checking
- [inspection.md](inspection.md) for one-run output capture and rendering
- [tracing.md](tracing.md) for one-run timing and collectors
- [benchmarking.md](benchmarking.md) for repeated-run measurement and sweeps

## Advanced Topics

- [regions.md](regions.md) for batching, scatter/gather, and custom region
  semantics
- [performance.md](performance.md) for practical guidance on batching,
  concurrency, and throughput tradeoffs
- [scaffolding.md](scaffolding.md) to wrap a new model inside an explicit
  pipeline

## Package-Owned Docs

- [../packages/core/docs/reference.md](../packages/core/docs/reference.md)
- [../packages/tensor/docs/reference.md](../packages/tensor/docs/reference.md)
- [../packages/vision/docs/reference.md](../packages/vision/docs/reference.md)
- [../packages/onnx/docs/reference.md](../packages/onnx/docs/reference.md)
- [../packages/torch/docs/reference.md](../packages/torch/docs/reference.md)

## Documentation Website

- [privacy.md](privacy.md) describes hosting, optional analytics, cookies,
  and privacy choices for the ml-pipes documentation sites.

## Community Packages

- [verified_community_packages.md](verified_community_packages.md) for the
  ownership, support, security, and release policy for verified integrations.
