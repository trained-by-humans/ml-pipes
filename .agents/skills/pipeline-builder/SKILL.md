---
name: pipeline-builder
description: Compose or improve concrete pipelines with ml-pipes. Use when Codex needs to start from a new or existing pipeline, inspect available operators, choose compositions, add missing local operators only when necessary, and get the pipeline working correctly first. Defer tracing, sweep, or benchmark optimization work until the pipeline already runs and validates, unless performance is the blocking issue.
---

# Pipeline Builder

Repository documentation Markdown files define semantics. This file only
drives pipeline composition decisions.

Use this skill when the task is to compose, adapt, extend, simplify, or
document a concrete pipeline in an example or downstream integration.

## Goal

The goal of this skill is to build a concrete working pipeline, not to extract
shared framework or package behavior.
Stop once the pipeline validates and behaves correctly on the provided
inputs and expected outputs, or the remaining issue needs debugging.

## Follow this Workflow

1. Understand the pipeline boundary.
   Define the pipeline input and output first. Start from the boundary you
   need to turn into the result.

2. Extract the important transformations.
   List the meaningful transformations needed to turn the input into the
   output.

3. Check existing examples.
   Identify the task category or domain, then check `examples/README.md` to
   see whether a similar pipeline already exists and use it as a reference.

4. Check existing operator packages.
   Check `docs/operators/README.md` for existing operators. Read
   `docs/OPERATORS.md` only when you need a deeper definition of what counts
   as an operator.

5. Generate any missing operators.
   If existing operators do not cover the pipeline, generate the missing
   functions or operators needed for this pipeline.
   Do not introduce new shared operators while building a pipeline. Prefer
   local functions or local operators until reuse requirements are confirmed.

6. Compose the pipeline.
   Follow `docs/COMPOSITION.md` to turn those transformations into an explicit
   pipeline.

7. Validate the composition.
   Run `validate()` to make sure the pipeline boundaries connect. Read
   `docs/VALIDATION.md` when validation fails or boundary contracts changed.

8. Run the pipeline on a representative input.
   If an input is provided, execute the pipeline to make sure it runs.

9. Compare against the expected output.
   If an expected output is provided, compare the result against it.

10. Inspect drift when the result does not match.
   Use `inspect()` to check which step starts to drift from the expected
   result.

## Hand Off To

- `pipeline-debugger` when the task is debugging an existing pipeline instead
  of building one, or when the generated pipeline does not work as expected
