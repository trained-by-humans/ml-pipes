---
name: pipeline-debugger
description: Debug broken or unexpected pipelines in ml-pipes. Use when Codex needs to reproduce a failing pipeline, reduce it to the smallest failing sub-pipeline, localize the first broken operator or boundary, classify the failure type, and gather evidence with describe, validate, inspect, tracing, or debug examples before deciding ownership or implementing the fix.
---

# Pipeline Debugger

Repository documentation Markdown files define semantics. This file only
drives failure localization.

Use this skill when a pipeline already exists but the first bad boundary,
step, or failure class is not yet known.

## Goal

The goal of this skill is localization, not repair.
Stop once the first confirmed failing boundary or root cause is identified and
the finding matches the user's report.

## Follow this Workflow

1. Identify the failing pipeline and input.
   Start from the code, command, test, factory input, or payload the user
   provided that matches the reported issue.
   If the failing pipeline code is missing or incomplete, use
   `examples/README.md` to find the closest runnable example or repro target.

2. Reproduce and reduce the issue.
   Reproduce the issue in the provided code path first.
   If that path is not runnable, reproduce it with the closest example, then
   reduce the failing case to the smallest pipeline that still shows the same
   problem. Read `docs/COMPOSITION.md` if the pipeline shape itself is unclear.

3. Validate composition first.
   Run `validate()` before deeper runtime debugging. Read `docs/VALIDATION.md`
   when boundary mismatch or composition failure is a likely cause.

4. Inspect the data flow.
   Use `inspect()` to see how the input is processed through the pipeline. Use
   `examples/run_inspect.py` and `examples/run_inspect_errors.py` as the first
   reference for localizing the bad step.

5. Trace only when runtime behavior matters.
   Use tracing only when inspection is not enough or the problem is about
   timing, runtime order, concurrency, or slow steps. Read `docs/TRACING.md`
   only when tracing is actually needed.

6. Stop at a confirmed localized issue.
   Finish once the issue is isolated and confirmed to match what the user
   described. The output of this skill should be a localized issue and root
   cause, not the final fix.

## Hand Off To

- `pipeline-builder` only when the user wants follow-up changes to the
  pipeline after the issue is localized

Otherwise, report the localized issue and root cause back to the user.
If the finding points to shared core or operator-package behavior, report that
explicitly and wait for the user to request a maintainer-side fix.
