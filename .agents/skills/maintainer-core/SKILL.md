---
name: maintainer-core
description: Maintain shared ml-pipes behavior in the core library. Use when Codex needs to change `src/ml_pipes/` runtime semantics, composition, validation, typing, tracing, benchmarking, CLI behavior, or shared tests and docs that affect more than one reusable operator workflow, example, or pipeline.
---

# Maintainer Core

Work in the core layer only when the issue is generic across multiple
pipelines, examples, or reusable operator workflows.

## Scope

- `src/ml_pipes/`
- `tests/`
- top-level docs that define shared behavior

Do not absorb example-specific or package-specific fixes into core.

## Workflow

1. Reproduce the problem with an existing failing test, a focused new test, or
   the exact user repro.
2. Read the matching shared docs before changing behavior:
   - `VALIDATION.md` for contract, strict, and inference semantics
   - `COMPOSITION.md` and `ARCHITECTURE.md` for composition, context, and
     runtime ownership
   - `TRACING.md`, `BENCHMARKING.md`, and `src/ml_pipes/__main__.py` for
     tracing, benchmarking, and CLI changes
3. Change the smallest shared surface that fixes the generic behavior.
4. Preserve the core design:
   - keep execution explicit and operator-list-driven
   - prefer composition over adding model-specific engine modes
   - keep model or task quirks out of core
   - keep reusable logic generic and operator-oriented
5. Verify in the core layer:
   - add or update focused tests near the changed behavior
   - cover both valid and failing paths when validation changes
   - cover strict or inference validation when those modes are involved
   - cover `run` or `benchmark` CLI paths when factory or CLI behavior changes
   - gather before and after tracing or benchmark evidence for performance work

## Required Checks

- Re-run `validate()` in any pipeline or example used to reproduce the issue.
- Keep docs in sync when user-facing semantics change.
- Prefer existing tests and minimal new repros over broad refactors during a
  fix.

## Read These Docs

- `ARCHITECTURE.md`
- `COMPOSITION.md`
- `VALIDATION.md`
- `TRACING.md`
- `BENCHMARKING.md`
- `README.md`
