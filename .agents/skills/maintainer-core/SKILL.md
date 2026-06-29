---
name: maintainer-core
description: Maintain shared ml-pipes behavior in the core library. Use when Codex needs to change `src/ml_pipes/` runtime semantics, composition, validation, typing, tracing, benchmarking, CLI behavior, or shared tests and docs that affect more than one operator package, example, or pipeline.
---

# Maintainer Core

Repository documentation Markdown files define semantics. This file only
drives scope and verification decisions.

Use this skill only when the request belongs in shared framework behavior or
shared framework-facing docs across more than one pipeline, example, or
operator package.

## Follow this Workflow

1. Confirm the framework surface.
   Use `README.md` to check the top-level framework surface and user-facing
   entry points.

2. Confirm the semantic intent.
   Use `docs/DESIGN.md` before touching shared behavior.

3. Locate the owning runtime surface.
   Use `docs/ARCHITECTURE.md` to find the right runtime surface in
   `src/ml_pipes/`.

4. Read subsystem docs only when needed.
   Read `docs/COMPOSITION.md`, `docs/VALIDATION.md`, `docs/TRACING.md`, or
   `docs/BENCHMARKING.md` only when that subsystem is part of the change.

5. Work only in core-owned surfaces.
   Keep the change in `src/ml_pipes/`, related tests, and shared docs.

6. Keep non-core behavior out of core.
   Keep model-specific behavior, example wiring, and one-off glue out of core.

## Hand Off When

- the change is really operator-package work -> `maintainer-operators`
- the change is local to one example or downstream pipeline -> ask the user
  before switching to `pipeline-builder`
- ownership is still unclear -> `maintainer-triage`
