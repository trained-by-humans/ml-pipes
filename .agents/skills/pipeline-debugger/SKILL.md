---
name: pipeline-debugger
description: Debug broken or unexpected pipelines in ml-pipes. Use when Codex needs to reproduce a failing pipeline, reduce it to the smallest failing sub-pipeline, localize the first broken operator or boundary, classify the failure type, and gather evidence with describe, validate, inspect, tracing, or debug examples before deciding ownership or implementing the fix.
---

# Pipeline Debugger

Focus on failure localization, not ownership.

- Find where the pipeline breaks.
- Explain what kind of failure it is.
- Produce evidence another skill can act on.

Do not decide final ownership by default. Hand off to `change-triage` when the
main remaining question is whether the fix belongs in core, reusable
operators, or a local pipeline.

## Workflow

1. Reproduce the exact failing command, example, test, factory invocation, or
   input first.
2. Reduce to the smallest failing pipeline, sub-pipeline, or region that still
   shows the same failure.
3. Confirm the actual structure before patching:
   - use `describe(show_defaults=True)` to see the current operator chain
   - use the exact example, test, or `python -m ml_pipes run` command that
     reproduces the failure
4. Validate before deep runtime debugging:
   - run `validate()` first
   - use `validate(strict=True)` when missing annotations or unresolved `Any`
     boundaries may be hiding the problem
   - use `validate(inference=True)` when backward inference may clarify the
     expected pipeline input or context contract
5. Inspect runtime behavior when validation alone is not enough:
   - run `inspect()` to find the first broken operator, region boundary, or
     context handoff
   - compare the last good step with the first bad step
   - prefer the first failure over downstream symptoms
6. Use tracing only when runtime order, concurrency, latency, or side effects
   matter. Do not jump to tracing before validation or inspection.
7. Classify the failure so the next skill inherits a clear diagnosis:
   - `composition`
   - `contract`
   - `context`
   - `factory or config`
   - `runtime data`
   - `region or concurrency`
   - `performance or regression`
8. Hand off based on the result:
   - `pipeline-builder` when the fix is local pipeline composition or a small
     local operator
   - `change-triage` when ownership across layers matters
   - maintainer skills when the evidence already points to core or reusable
     operator work

## Required Checks

- Reproduce before patching.
- Prefer the smallest failing repro over the full original pipeline.
- Run `validate()` before ad hoc print-debugging.
- Use `inspect()` to localize the first bad step when the pipeline runs but
  behaves incorrectly.
- Keep ownership decisions out of this skill unless the evidence is already
  obvious.

## Read These Docs

- `README.md`
- `VALIDATION.md`
- `COMPOSITION.md`
- `TRACING.md`
- `examples/run_inspect.py`
- `examples/run_inspect_errors.py`
