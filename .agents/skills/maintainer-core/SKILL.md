---
name: maintainer-core
description: Maintain shared ml-pipes framework behavior. Use when Codex needs to work on framework-owned runtime, tooling, or shared docs implemented under `packages/core/` and change pipeline execution, generic operators, context or region behavior, composition, validation, typing, tracing, inspection, benchmarking, collectors, factory, CLI behavior, or shared docs/tests.
---

# Maintainer Core

Repository documentation Markdown files define semantics. This file only
drives scope and verification decisions.

Use this skill only when the request has already been routed to core-owned
framework behavior or shared framework-facing docs.

## Goal

The goal of this skill is to change shared framework behavior and
framework-owned docs. `packages/core/` is the implementation home of the
framework engine and tooling, not just another package surface.
Do not absorb package-specific or example-specific logic here.
Stop once the framework change is implemented and verified in core-owned
surfaces.

## Follow This Workflow

1. Confirm framework ownership.
   Read `docs/PACKAGES.md`, `packages/core/README.md`, and
   `packages/core/docs/INDEX.md` to confirm the routed target really belongs
   to the shared framework layer rather than to one package-owned domain
   surface.

2. Confirm the semantic intent.
   Read `docs/DESIGN.md` before touching shared behavior.

3. Locate the owning framework surface.
   Read `docs/ARCHITECTURE.md`, then locate the owner under
   `packages/core/src/ml_pipes/`.
   Use the existing architecture sections as the ownership map:
   - `Pipeline`
   - `Operators`
   - `Context`
   - `Regions`
   - `Composition`
   - `Validation`
   - `Tracing`
   - `Inspection`
   - `Benchmarking`
   - `Pipeline Description`

4. Read subsystem docs only when needed.
   Read `docs/COMPOSITION.md`, `docs/VALIDATION.md`, `docs/TRACING.md`, or
   `docs/BENCHMARKING.md` only when that subsystem is part of the change.

5. Confirm framework fit.
   If the docs show the change does not fit the shared framework layer handled
   by core, reject it for this scope and report why it does not fit core.

6. Implement the requested change.
   Once the change still fits the shared framework layer after that check,
   implement the change under `packages/core/src/ml_pipes/` in the file that
   matches the behavior already grouped there.

7. Check the implementation against the shared operator rules when needed.
   If the change adds, changes, or updates a generic reusable operator in
   `ml_pipes.standard`, read `docs/OPERATORS.md` and make sure the
   implementation follows those rules.

8. Verify the framework behavior.
   Run the smallest targeted shared tests or examples that confirm the new
   framework behavior and the final public shape of the change.

9. Align the docs to the verified surface.
   If the verified framework change affects a documented public surface,
   update the relevant core package docs and any shared framework docs that
   describe that surface.

## Output

- report which framework surface owns the change
- report whether an existing framework surface already matched, an existing
  surface was extended, or a new framework surface was added
- report the focused verification result

## Reject When Scope Does Not Fit

- the change is really package-owned surface work; explain why it does not
  fit core
- the change is local to one example or downstream pipeline; explain why it
  does not fit core
