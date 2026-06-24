---
name: pipeline-builder
description: Compose or improve concrete pipelines with ml-pipes. Use when Codex needs to start from a new or existing pipeline, inspect available operators, choose compositions, add missing local operators only when necessary, and get the pipeline working correctly first. Defer tracing, sweep, or benchmark optimization work until the pipeline already runs and validates, unless performance is the blocking issue.
---

# Pipeline Builder

Stay inside ml-pipes as much as possible. Prefer existing operators and
composition before generating new code.

## Phase Rule

Build for correctness first.

- The first goal is a working, understandable, validated pipeline.
- Do not optimize during initial construction unless performance is the reason
  the pipeline is failing, unusable, or explicitly requested.
- Treat optimization as a later phase that begins only after the pipeline runs
  correctly on a stable repro, example, or factory input.

## Workflow

1. Start from the closest existing pipeline, example, or factory instead of
   building from zero when a nearby pattern already exists.
2. Search available operators before writing new ones:
   - read `docs/OPERATORS.md`
   - search examples, tests, and `src/ml_pipes/`
   - reuse existing reusable operators when they already cover the need
3. Compose the pipeline in stages:
   - list the required transformations and boundaries
   - map each stage to existing operators first
   - use local callables or local operators only for the missing pieces
   - keep missing logic local until reuse across pipelines is clear
4. Validate continuously:
   - call `validate()` after every pipeline mutation or composition change
   - use `Pipeline(..., auto_validate=True)` while iterating when fail-fast
     construction is useful
   - use `describe(show_defaults=True)` to confirm the current structure
5. Debug for correctness with existing tools:
   - use `inspect()` when intermediate values or failing steps are unclear
   - use tracing to understand step timing and runtime behavior
   - use `python -m ml_pipes run` for reproducible factory-based execution
   - hand off to `pipeline-debugger` when the first broken boundary or failure
     class is not obvious
6. Stop the initial build phase once the pipeline:
   - runs on the target repro, example, or factory input
   - validates after the latest composition changes
   - is understandable through `describe()` and reproducible through a command,
     example, or test
7. Only then begin performance work when needed:
   - capture a baseline first
   - use tracing to identify hotspots before changing structure
   - use `Benchmark`, `BenchmarkBuilder`, sweeps, or `python -m ml_pipes benchmark`
     when comparing alternatives or optimizing
   - re-validate after every optimization change
8. Escalate ownership when needed:
   - route to `pipeline-debugger` first when a broken pipeline needs failure
     localization more than immediate code changes
   - route to `maintainer-operators` if a missing operator is reusable
   - route to `maintainer-core` if the blocker is shared runtime behavior
   - route to `change-triage` when the pipeline behavior is understood but the
     owning layer is still unclear

## Required Checks

- Prefer composition over inheritance and reuse over regeneration.
- Validate after every pipeline change.
- Do not benchmark by default during first-pass pipeline construction.
- Benchmark after optimization work or when choosing between competing
  compositions in the later performance phase.
- Verify new operators in a pipeline, not only as isolated functions.

## Read These Docs

- `README.md`
- `docs/OPERATORS.md`
- `docs/COMPOSITION.md`
- `docs/VALIDATION.md`
- `docs/TRACING.md`
- `docs/BENCHMARKING.md`
- `docs/PERFORMANCE.md`
