---
name: maintainer-operators
description: Add or maintain operators in ml-pipes operator packages. Use when Codex needs to check for an existing exact or similar operator, decide whether to extend it or add a new one in the right package or file, validate that the boundary is truly generic for that package, implement it using the operator rules, and verify it with a small runnable example.
---

# Maintainer Operators

Repository documentation Markdown files define semantics. This file only
drives operator-layer decisions.

Use this skill when the task is to add, change, simplify, or document an
operator in a shared operator package.

## Goal

The goal of this skill is to place or update a truly generic operator in the
right shared package, not to push one-off pipeline logic into shared code.
Stop once package fit, operator shape, and small-example verification are
confirmed.

## Follow this Workflow

1. Confirm package fit.
   If the package is already specified, verify the operator belongs there.
   Otherwise, find the most relevant operator package or file.

2. Check for an existing exact operator.
   Look for an operator whose semantics already match the requested behavior.
   If the existing operator already matches the request, skip the change and
   report the existing operator instead.

3. Check for a semantically similar operator.
   If a nearby operator covers most of the requested behavior, decide whether
   extending that operator still makes semantic sense and preserves single
   responsibility.
   Prefer extending an existing operator in place over creating a duplicate
   operator with slightly different semantics.

4. Confirm boundary fit.
   If the request still needs a new or changed operator, check that the
   operator boundary is truly generic and matches the domain of that package
   and the other operators already in it.

5. Validate the operator shape.
   Use `docs/OPERATORS.md` to make sure the operator follows the shared rules:
   clear boundary, precise `__call__` annotations, meaningful config, and
   `resolve_contract(...)` only when needed.

6. Place or update the operator.
   Update the existing operator in place when the request fits it.
   Create a new operator only when the behavior does not fit an existing one.
   Do not move an operator unless the user explicitly requested that move.

7. Verify with a small runnable example.
   Create and run a focused example that checks how the operator composes in a
   pipeline and how it behaves with the operator tooling from
   `docs/OPERATORS.md`, such as composition, validation, description,
   inspection, and tracing when useful.

## Report Back

- report whether the operator belonged in the specified or selected package
- report whether an existing operator already matched, an existing operator was
  extended, a new operator was added, or the request was rejected as not
  generic enough for a shared package
- report the verification result from the small example
