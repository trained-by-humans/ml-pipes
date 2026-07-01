---
name: maintainer-triage
description: Triage proposed changes in ml-pipes once the requested behavior is understood. Use when Codex needs to decide whether a change, or each component of a mixed change, should stay local to an example or downstream pipeline, move into an operator package, or belong in the core library.
---

# Maintainer Triage

Repository documentation Markdown files define semantics. This file only
drives ownership decisions.

Use this skill when the requested outcome is understood and the next decision
is which layer should own the change.

## Goal

The goal of this skill is ownership, not implementation.
For mixed changes, split the work into components and triage each component
separately.
Start from the most local scope and move inward only when a component clearly
belongs in a shared layer.
Stop once the correct owning scope is clear.

## Check Scopes In Order

1. Check `examples/`
   - Read `examples/README.md`.
   - Confirm whether the current component is local to one example or
     downstream pipeline.
   - If yes, stop triage for that component and return to the skill router to
     choose `pipeline-builder` or `pipeline-debugger`.
   - If no, continue to the next scope.

2. Check operator packages
   - Read `docs/OPERATORS.md` and `docs/operators/README.md`.
   - Confirm whether the current component belongs in an operator package.
   - If yes, stop triage for that component and return to the skill router to
     choose `maintainer-operators`.
   - If no, continue to the next scope.

3. Check core framework
   - Read `docs/DESIGN.md` and `docs/ARCHITECTURE.md`.
   - Confirm whether the current component belongs in shared runtime or
     tooling behavior.
   - If yes, stop triage for that component and return to the skill router to
     choose `maintainer-core`.
   - If no, use the fallback below.

## If Ownership Is Still Unclear

- Ask the user whether the current component is meant to stay local to one
  example or integration, be reused across multiple pipelines, or change
  shared framework behavior.
- If that is still unclear, prefer keeping the component local rather than
  moving it into a shared layer.
