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

5. Confirm core fit.
   If the core docs show the change does not fit the core package, reject it
   for this scope and report why it does not fit core.

6. Implement the requested change.
   Once the change still fits core after that check, implement the change under
   `packages/core/src/ml_pipes/` in the file that matches the
   behavior already grouped there.

7. Check the implementation against the shared operator rules when needed.
   If the change adds, changes, or updates a generic reusable operator in
   `ml_pipes.standard`, read `docs/OPERATORS.md` and make sure the
   implementation follows those rules.

8. Verify the core behavior.
   Run the smallest targeted shared tests or examples that confirm the new
   core behavior and the final public shape of the change.

9. Align the docs to the verified surface.
   If the verified core change affects a documented public surface, update
   the relevant core package docs and any shared framework docs that describe
   that surface.

## Output

- report which core surface owns the change
- report whether an existing core surface already matched, an existing
  surface was extended, or a new core surface was added
- report the focused verification result

## Reject When Scope Does Not Fit

- the change is really package-owned surface work; explain why it does not
  fit core
- the change is local to one example or downstream pipeline; explain why it
  does not fit core
