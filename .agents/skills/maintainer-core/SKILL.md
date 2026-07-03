---
name: maintainer-core
description: Maintain shared ml-pipes behavior in the core package. Use when Codex needs to work inside an already-routed core target under `packages/core/` or shared `docs/` and change runtime semantics, generic operators, validation, typing, tracing, inspection, benchmarking, collectors, factory, CLI behavior, or shared docs/tests.
---

# Maintainer Core

Repository documentation Markdown files define semantics. This file only
drives scope and verification decisions.

Use this skill only when the request has already been routed to core-owned
framework behavior or shared framework-facing docs.

## Goal

The goal of this skill is to change shared framework behavior in
`packages/core/`, not to absorb package-specific or example-specific logic.
Stop once the core change is implemented and verified in core-owned surfaces.

## Follow This Workflow

1. Confirm the routed core target.
   Read `docs/PACKAGES.md`, `packages/core/README.md`, and
   `packages/core/docs/INDEX.md` to confirm the routed target really belongs
   to the core package.

2. Confirm the semantic intent.
   Read `docs/DESIGN.md` before touching shared behavior.

3. Locate the owning runtime surface.
   Read `docs/ARCHITECTURE.md`, then locate the owner under
   `packages/core/src/ml_pipes/`.
   Typical owners include:
   - pipeline composition and dispatch
   - generic reusable operators in `ml_pipes.standard`
   - validation and typing helpers
   - tracing, collectors, inspection, and benchmarking
   - factory and CLI behavior

4. Read subsystem docs only when needed.
   Read `docs/COMPOSITION.md`, `docs/VALIDATION.md`, `docs/TRACING.md`, or
   `docs/BENCHMARKING.md` only when that subsystem is part of the change.

5. Keep the change in core-owned surfaces.
   Work in `packages/core/src/ml_pipes/`, shared docs under `docs/`, and
   shared tests under `tests/`.

6. Keep non-core behavior out of core.
   Do not move package-specific payloads, runtime boundaries, or task-specific
   postprocess into core. Those belong in the owning package under
   `packages/<name>/`.

7. Update user-facing ownership docs when needed.
   If the public core surface changes, update the relevant core package docs
   and any shared framework docs that describe that surface.

8. Verify at the framework level.
   Run the smallest targeted shared tests or examples that confirm the new
   core behavior and any affected public surface.

## Return To The Router When

- the change is really package-owned surface work; report the corrected
  target package, module, file, and line when known
- the change is local to one example or downstream pipeline; report the
  corrected target under `examples/**`
- the correct target is still unclear; report the unresolved alternatives
