---
name: maintainer-triage
description: Triage proposed changes in ml-pipes once the requested behavior is understood. Use when Codex needs to decide whether a change should stay local to an example or downstream pipeline, move into an operator package, or belong in the core library.
---

# Maintainer Triage

Repository documentation Markdown files define semantics. This file only
drives ownership decisions.

Use this skill when the requested outcome is understood and the next decision
is which layer should own the change.

Anchor the task in one concrete target: test, API surface, docs page,
example, or command.

If the result would switch between `integrator` and `maintainer` roles, ask
the user before handing off.

## Check Scopes In Order

1. Check `examples/`
   - Read `examples/README.md`.
   - Confirm whether the change is local to one example or downstream
     pipeline.
   - If yes, hand off to `pipeline-builder` or `pipeline-debugger` and work in
     `examples/`.
   - If no, continue to the next scope.

2. Check operator packages
   - Read `docs/OPERATORS.md` and `docs/operators/README.md`.
   - Confirm whether the change belongs in an operator package.
   - If yes, hand off to `maintainer-operators` and work in the
     operator-facing surfaces under `src/ml_pipes/`.
   - If no, continue to the next scope.

3. Check core framework
   - Read `docs/DESIGN.md` and `docs/ARCHITECTURE.md`.
   - Confirm whether the change belongs in shared runtime or tooling
     behavior.
   - If yes, hand off to `maintainer-core` and work in `src/ml_pipes/` plus
     shared docs/tests.
   - If no, use the fallback below.

## If Ownership Is Still Unclear

- Ask the user whether the change is meant to stay local to one example or
  integration, be reused across multiple pipelines, or change shared
  framework behavior.
- If that is still unclear, prefer keeping the change local rather than
  moving it into a shared layer.
