---
name: pipeline-debugger
description: Debug broken or unexpected pipelines in ml-pipes. Use when Codex needs to reproduce a failing pipeline, reduce it to the smallest failing sub-pipeline, localize the first bad boundary or step, and identify the concrete follow-up target such as package, module, file, and line.
---

# Pipeline Debugger

Repository documentation Markdown files define semantics. This file only
drives failure localization.

Use this skill when a pipeline already exists but the first bad boundary,
step, or concrete follow-up target is not yet known.

## Goal

The goal of this skill is localization, not repair.
Stop once the first confirmed failing boundary or root cause is identified and
reduced to concrete follow-up targets.

## Follow This Workflow

1. Identify the failing pipeline and input.
   Start from the code, command, test, factory input, or payload the user
   provided.
   If the exact path is missing or incomplete, use `examples/README.md` to
   find the closest runnable example or repro target.

2. Reproduce and reduce the issue.
   Reproduce the reported failure in the provided path first.
   If that path is not runnable, reproduce it with the closest example, then
   reduce the failing case to the smallest pipeline that still shows the same
   problem.

3. Validate composition first.
   Run `validate()` before deeper runtime debugging.
   Read `docs/VALIDATION.md` when boundary mismatch or contract failure is a
   likely cause.

4. Inspect the value flow.
   Use `inspect()` to localize the first step whose output no longer matches
   expectations.
   Use `examples/run_inspect.py` and `examples/run_inspect_errors.py` as the
   first reference for inspection-driven debugging.

5. Trace only when runtime behavior matters.
   Use tracing only when inspection is not enough or the problem is about
   timing, runtime order, concurrency, or slow steps.
   Read `docs/TRACING.md` only when tracing is actually needed.

6. Record the concrete follow-up targets.
   For the first confirmed failure, identify the narrowest concrete targets
   you can support from the evidence:
   - package
   - module
   - file
   - line
   If the exact line is still unclear, stop at the narrowest confident target.

7. Finish with the next target.
   Stop once the localized issue, root cause, and concrete targets are
   clear enough for the next routing step.

## Output

Report back:

- the localized failure or root cause
- the concrete follow-up target: package, module, file, and line when known
- any remaining uncertainty that blocks narrowing the target further

## Switch To

- `pipeline-builder` only when the user wants the current example or
  downstream pipeline fixed and the issue is now concrete enough for local
  pipeline repair
- if a local fix alone is not possible, report the concrete target instead of
  switching to `pipeline-builder`
