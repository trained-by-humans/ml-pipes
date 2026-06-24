# ml-pipes Guidance

## First Classify The Request

Every task that touches pipelines, operators, validation, tracing, or examples
must first be classified as one of these layers:

- `core library`
- `reusable operators`
- `local example or external pipeline`

Do not edit core when the issue is specific to one example or one downstream
pipeline. Do not promote example-specific logic into a reusable operator until
reuse is justified or the user explicitly asks for that promotion.

## Layer Ownership

### Core Library

Generic execution, composition, validation, typing, tracing, benchmarking, and
generic operator semantics belong in `src/ml_pipes/`.

### Reusable Operators

Reusable operator work belongs in the core library or in a dedicated operator
layer when the logic is reusable across more than one local pipeline.

### Local Examples And External Pipelines

Example wiring, one-off model quirks, project-specific glue, and local
operators should stay with the example or downstream pipeline until reuse is
clear.

## Operator And Pipeline Rules

- Prefer composition over inheritance.
- Prefer existing operators and pipeline composition before generating a new
  operator.
- New operator generation is the exception, not the default.
- Keep operators explicit and composable. Prefer small single-purpose operators
  over large model-specific blocks.
- Keep generic operators precision-agnostic unless the constraint is inherent
  to a runtime boundary.
- When converting imperative logic into pipelines, move incrementally and note
  what still resists composition.

## Validation And Verification

- Validate after every pipeline mutation or composition change.
- Use `Pipeline(..., auto_validate=True)` when fail-fast construction behavior
  is appropriate.
- When validation behavior is the subject, read `docs/VALIDATION.md` before changing
  code.
- When composition or context behavior is the subject, read
  `docs/COMPOSITION.md` and `docs/ARCHITECTURE.md`.
- When optimization work is proposed, do not stop at “it should be faster”;
  gather before/after evidence with tracing or benchmarking.

## Change Work Style

- Reproduce the reported behavior first from the exact example, test, or user
  repro when the request is about a bug, regression, or unexpected output.
- For features, enhancements, or refactors, anchor the request in the exact
  example, test, or command that should demonstrate the intended change.
- Gather evidence in increasing cost order: code search, `describe()`,
  `validate()`, `inspect()`, tracing, then benchmarks.
- Fix at the owning layer. Do not push a local example problem into core just
  because core is reachable.
- Add or update tests at the same layer that owns the fix.

## Skill Boundary

Keep durable repo-wide rules here. Put workflow-specific instructions in
`.agents/skills/`.

## Skill Router

When a task matches one of the workflows below, read the corresponding
`.agents/skills/.../SKILL.md` file before taking action.

- `pipeline-builder` -> `.agents/skills/pipeline-builder/SKILL.md`
  Use for creating or repairing a concrete pipeline. Stay correctness-first and
  defer optimization until the pipeline already works.
- `pipeline-debugger` -> `.agents/skills/pipeline-debugger/SKILL.md`
  Use when an existing pipeline is broken and the first failing step, boundary,
  or failure class is still unclear.
- `change-triage` -> `.agents/skills/change-triage/SKILL.md`
  Use when the behavior or requested change is understood and the main
  question is ownership: core library, reusable operators, or local
  example/external pipeline.
- `maintainer-core` -> `.agents/skills/maintainer-core/SKILL.md`
  Use for shared runtime, validation, composition, tracing, benchmarking, CLI,
  and shared test or doc changes in `src/ml_pipes/`.
- `maintainer-operators` -> `.agents/skills/maintainer-operators/SKILL.md`
  Use for reusable operator work: duplicate checks, new operators, operator
  maintenance, and pipeline-level verification of operator changes.

If more than one route applies, prefer this order:

- concrete pipeline work -> `pipeline-builder`
- broken pipeline with unclear failure -> `pipeline-debugger`
- ownership decision after debugging -> `change-triage`
- shared runtime fix -> `maintainer-core`
- reusable operator fix -> `maintainer-operators`
