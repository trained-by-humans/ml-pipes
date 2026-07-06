---
name: maintainer-operators
description: Add or maintain package-owned surfaces in ml-pipes packages. Use when Codex needs to work inside an already-routed package target, check whether a requested operator or package export already exists, update the right package-local code and docs, and verify the change with a focused pipeline or package-level check.
---

# Maintainer Operators

Repository documentation Markdown files define semantics. This file only
drives package-surface decisions.

Use this skill when the task is to add, change, simplify, or document a
shared surface owned by one of the package modules such as
`ml_pipes.tensor`, `ml_pipes.vision`, `ml_pipes.onnx`, or `ml_pipes.torch`
after the target package is already known.

## Goal

The goal of this skill is to place or update a truly reusable package-owned
surface in the right package, not to push one-off pipeline logic into shared
code.
Stop once package fit, surface shape, implementation, focused verification,
and package-doc alignment are complete.

## Follow This Workflow

1. Confirm the routed package target.
   Read `docs/PACKAGES.md`, `packages/<name>/README.md`, and
   `packages/<name>/docs/INDEX.md` to confirm the routed public module
   really belongs to the selected package.

2. Confirm package fit.
   If the package docs show the change does not fit this package, reject it
   for this scope and report why it does not fit this package.

3. Check for an exact existing surface.
   Look for an existing exact surface first: operator, value type, export,
   alias, or package doc entry.
   If it already matches the request, stop and report the existing surface.

4. Check for a semantically similar surface.
   If a nearby operator or exported value covers most of the request, decide
   whether extending it still preserves one clear package-owned boundary.
   Prefer extending an existing package surface over creating a duplicate.

5. Implement the requested change.
   After the package is clear, implement the change under
   `packages/<name>/src/ml_pipes/<module>/...` in the file that matches the
   behavior already grouped there.

6. Check the implementation against the shared operator rules when needed.
   If the change adds, changes, or updates an operator, read
   `docs/OPERATORS.md` and make sure the implementation follows those rules.

7. Place the public package surface.
   If the public package surface changed, update the package `__init__`
   exports.

8. Verify with a focused package-level check.
   Use the smallest pipeline, package test, or example that proves the updated
   surface composes correctly and matches the final public shape.
   If the public package surface changed, include the narrowest curated
   surface tests that confirm the export shape.

9. Align package docs to the verified surface.
   If the verified package surface changed, update
   `packages/<name>/README.md` and `packages/<name>/docs/INDEX.md` to
   describe that final public surface.

## Output

- report which public module and package own the change
- report whether an existing surface already matched, an existing surface was
  extended, or a new package surface was added
- report the focused verification result

## Reject When Scope Does Not Fit

- the routed package target does not fit; explain why it does not fit this
  package
