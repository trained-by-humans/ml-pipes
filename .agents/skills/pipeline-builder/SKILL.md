---
name: pipeline-builder
description: Compose or improve concrete pipelines with ml-pipes. Use when Codex needs to start from a new or existing pipeline, identify the package chain it should use, select current package-owned operators, add missing local steps only when necessary, and get the concrete pipeline working before considering shared extraction.
---

# Pipeline Builder

Repository documentation Markdown files define semantics. This file only
drives concrete pipeline composition decisions.

Use this skill when the task is to compose, adapt, extend, simplify, or
document a concrete pipeline in an example or downstream integration.

## Goal

The goal of this skill is to build a concrete working pipeline, not to extract
shared framework or package behavior.
Stop once the pipeline validates and behaves correctly on representative
inputs and expected outputs, or the remaining issue needs debugging.

## Follow This Workflow

1. Define the boundary first.
   State the pipeline input and output before selecting operators.

2. Identify the package chain.
   Read `docs/PACKAGES.md` and decide which package surfaces the pipeline
   should cross.
   Common chains include:
   - `vision -> onnx -> tensor -> vision`
   - `vision -> tensor -> torch`
   - `core + standard` around any of the above

3. Choose the main guide.
   Use:
   - `docs/SCAFFOLDING.md` for model wrapping and runtime scaffolding
   - `docs/COMPOSITION.md` for general pipeline composition

4. Check runnable examples first.
   Read `examples/README.md` and start from the closest example or repro
   target before inventing a new pipeline shape.

5. Inspect the owning package surfaces.
   For each package in the chain, read the package `README.md` and
   `docs/INDEX.md` to find the current exported operators and value types.

6. List the meaningful transformations.
   Break the pipeline into explicit stages and map each stage to the package
   that should own it.

7. Add local steps only when needed.
   If existing package surfaces do not cover a stage, add the missing local
   functions or local operators in the example or downstream code.
   Use `docs/OPERATORS.md` only when you need guidance for a local operator.
   Do not introduce new shared package or core behavior while building the
   pipeline.

8. Compose and validate.
   Build the explicit pipeline and run `validate()`.
   Read `docs/VALIDATION.md` when boundary mismatch or contract issues appear.

9. Run the pipeline on a representative input.
   Execute the pipeline on a realistic input and compare against the expected
   output when one is available.

10. Inspect drift when needed.
    Use `inspect()` to localize where the value begins to drift from the
    expected result.

11. Record the concrete follow-up targets when local work stops.
    If the remaining issue is not local composition work, identify the
    narrowest concrete targets you can support from the current evidence:
    package, module, file, and line when known.
    Report those targets so the next routing step can continue.

## Switch To

- `pipeline-debugger` only when the user wants the local pipeline failure
  localized further before making changes
