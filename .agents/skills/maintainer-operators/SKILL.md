---
name: maintainer-operators
description: Add or maintain operators in ml-pipes operator packages. Use when Codex needs to place a new operator in the right package or file, check that its boundary is truly generic for that package, implement it using the operator rules, and verify it with a small runnable example.
---

# Maintainer Operators

Repository documentation Markdown files define semantics. This file only
drives operator-layer decisions.

Use this skill when the task is to add, change, simplify, or document an
operator in a shared operator package.

## Follow this Workflow

1. Confirm package fit.
   If the package is already specified, verify the operator belongs there.
   Otherwise, find the most relevant operator package or file.

2. Confirm boundary fit.
   Check that the operator boundary is truly generic and matches the domain of
   that package and the other operators already in it.

3. Validate the operator shape.
   Use `docs/OPERATORS.md` to make sure the operator follows the shared rules:
   clear boundary, precise `__call__` annotations, meaningful config, and
   `resolve_contract(...)` only when needed.

4. Place or update the operator.
   Add the operator to the selected package or update the existing one in
   place.

5. Verify with a small runnable example.
   Create and run a focused example that checks how the operator composes in a
   pipeline and how it behaves with the operator tooling from
   `docs/OPERATORS.md`, such as composition, validation, description,
   inspection, and tracing when useful.

## Report Back

- report whether the operator belonged in the specified or selected package
- report whether the operator was added, updated, or rejected as not generic
  enough for a shared package
- report the verification result from the small example
