---
name: change-triage
description: Triage proposed changes in ml-pipes once the requested behavior is understood. Use when Codex needs to decide whether a bug fix, regression fix, feature, enhancement, or refactor belongs in the core library, reusable operators, or a local example or external pipeline. If the first task is to localize a broken step or boundary, use `pipeline-debugger` first.
---

# Change Triage

Classify the change before changing code:

- `core library`
- `reusable operators`
- `local example or external pipeline`

Implement only in the owning layer and verify there.

Use this skill once the requested behavior is understood well enough that
ownership is the next question. If the pipeline is broken but the first
failing operator, boundary, or failure class is still unclear, hand off to
`pipeline-debugger` first.

## Workflow

1. Ground the change in a concrete repro first:
   - for bugs or regressions, reproduce the exact failing example, test, or
     command
   - for features, enhancements, or refactors, identify the smallest example,
     test, or command that should demonstrate the intended behavior
2. Gather evidence in increasing cost order:
   - search operators, tests, examples, and docs
   - inspect structure with `pipeline.describe(show_defaults=True)` when
     composition is unclear
   - run `pipeline.validate(...)` after any pipeline mutation and whenever
     ownership depends on type or context contracts
   - run `pipeline.inspect(...)` when step-level runtime behavior matters
   - use tracing for latency or step timing evidence
   - use benchmarks or sweeps for performance regressions or optimization
     proposals
3. Decide ownership:
   - `core library`: generic composition, typing, validation, tracing,
     benchmarking, CLI behavior, or runtime semantics
   - `reusable operators`: reusable operator behavior or shared operator-layer
     integration
   - `local example or external pipeline`: app-specific wiring, local
     operators, model quirks, sample code, or one-off glue
4. Fix only at the owning layer. Do not move a local workaround into core just
   because core is reachable.
5. Verify at the same layer:
   - core: focused tests plus relevant validation or inspection coverage
   - reusable operators: operator tests plus pipeline-level validation
   - local pipeline or example: rerun the exact repro or requested scenario,
     then validate and benchmark when performance is involved
6. Summarize the repro or target scenario, evidence, ownership decision, fix
   location, and verification steps.

## Required Checks

- Call `validate()` after every pipeline composition change.
- When optimizing, collect before and after evidence with tracing or benchmark
  results. Prefer existing benchmark helpers or `python -m ml_pipes benchmark`
  over ad hoc timers.
- Prefer existing operators and composition before creating a new operator.

## Read These Docs

- `README.md` for entry points, examples, and CLI discovery.
- `docs/VALIDATION.md` for contract, strict, and inference behavior.
- `docs/COMPOSITION.md` and `docs/ARCHITECTURE.md` for ownership and composition
  questions.
- `docs/OPERATORS.md` when deciding whether an existing operator already covers the
  use case.
- `docs/TRACING.md` for debugging and latency evidence.
- `docs/BENCHMARKING.md` and `docs/PERFORMANCE.md` for optimization and benchmark
  workflows.
