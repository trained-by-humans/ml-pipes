---
name: maintainer-operators
description: Add or maintain reusable operators in ml-pipes. Use when Codex needs to check whether an operator already exists, decide whether logic belongs in core versus reusable operator code versus a local pipeline, implement a new operator with correct annotations, and verify the operator inside a real pipeline.
---

# Maintainer Operators

Treat reusable operator work as a separate layer from both core runtime changes
and local pipeline glue.

## Workflow

1. Classify ownership first:
   - core library if the behavior is generic runtime, validation, composition,
     tracing, or benchmarking logic
   - reusable operator layer if the operator is reusable across multiple
     pipelines in the same domain
   - local pipeline if the logic is one-off, project-specific, or still
     exploratory
2. Check for duplicates before adding anything:
   - search `docs/operators/README.md`, `docs/OPERATORS.md`, `README.md`,
     `src/ml_pipes/`, tests, examples, and the target package
   - prefer an existing operator or pipeline composition if the behavior is
     already expressible
   - do not add a new operator when the need is really a short local callable
     or a pipeline wiring change
3. When adding a new reusable operator:
   - keep it stateless and single-purpose
   - keep it model-agnostic unless the package boundary is intentionally
     domain-specific
   - add precise `__call__` type annotations so validation can reason about it
   - keep precision constraints or runtime quirks localized to true boundaries
4. Verify at the package layer:
   - add operator-focused tests
   - validate the operator inside a pipeline, not only in isolation
   - use `inspect()` or tracing when the runtime path is unclear
   - benchmark only when the operator changes performance-sensitive code
5. Promote or demote ownership when evidence changes:
   - move to core only when the behavior is truly generic
   - keep it local if reuse never materializes

## Required Checks

- Search for existing operators before creating a new one.
- Call `validate()` on a pipeline that exercises the new or changed operator.
- Prefer composition of small operators over large fused operators.

## Read These Docs

- `docs/OPERATORS.md`
- `docs/operators/README.md`
- `README.md`
- `docs/VALIDATION.md`
- `docs/COMPOSITION.md`
- `docs/ARCHITECTURE.md`
- `docs/TRACING.md`
