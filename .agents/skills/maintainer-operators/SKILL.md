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
Stop once package fit, operator shape, package docs, and focused verification
are complete.

## Follow This Workflow

1. Confirm the routed package target first.
   Read `docs/PACKAGES.md` to confirm the routed public module and package.

2. Read the owning package docs.
   Read `packages/<name>/README.md` and `packages/<name>/docs/INDEX.md` for
   the selected package.
   Use `docs/OPERATORS.md` only for shared operator rules.

3. Check the current package surface.
   Look for an existing exact surface first: operator, value type, export,
   alias, or package doc entry.
   If it already matches the request, stop and report the existing surface.

4. Check for a semantically similar surface.
   If a nearby operator or exported value covers most of the request, decide
   whether extending it still preserves one clear package-owned boundary.
   Prefer extending an existing package surface over creating a duplicate.

5. Confirm package fit.
   Use package ownership, not convenience, to decide where the change belongs:
   - generic framework and generic operators -> core
   - shared NumPy tensor work -> tensor
   - image preprocessing, typed vision outputs, rendering/logging -> vision
   - ONNX runtime boundary -> onnx
   - Torch execution boundary and Torch-native postprocess -> torch
   If the routed package does not fit, stop and return the corrected concrete
   target to the skill router.

6. Choose the package-local source file.
   After the package is clear, place the change under
   `packages/<name>/src/ml_pipes/<module>/...` in the file that matches the
   behavior already grouped there.

7. Keep package docs and exports aligned.
   If the public package surface changes, update the package `__init__`,
   package README, package docs index, and any shared tests that curate the
   public surface.

8. Verify with a focused package-level check.
   Use the smallest pipeline, package test, or example that proves the updated
   surface composes correctly and is exported/documented correctly.

## Report Back

- report which public module and package own the change
- report whether an existing surface already matched, an existing surface was
  extended, or a new package surface was added
- report the focused verification result

## Return To The Router When

- the routed package target does not fit; report the corrected package,
  module, file, and line when known
