# ml-pipes Repository Guide

## What ml-pipes Is Semantically

`ml-pipes` is a framework for composing ML systems around explicit data flow,
where:
- The primary artifact is the pipeline: an ordered sequence of operators that
  move data from one step to the next.
- The operator is the unit of composition; each operator owns one meaningful
  boundary.

## What ml-pipes Is Mechanically

`ml-pipes` is a framework that:

- Allows integration from the source or from published packages.
- Follows the BYOC pattern, so it can be used in any computation-centric
  domain.
- Offers official and community operators packaged by domain.
- Creates value by providing the harness around pipelines:
  Validation, inspection, tracing, and benchmarking all support the same goal.

## Source Of Truth

- Repository documentation Markdown files are the source of truth for all semantics and APIs:
  `README.md`, `docs/**/*.md`, and `examples/**/*.md`.
- Agent files should only be used to choose the right workflow and the target layer for any changes.
- If there is a conflict between repository documents and any agent or skill file,
  follow the repository documentation.

## Repository Structure

- `src/ml_pipes/`: framework runtime, shared operators, typing, validation,
  tracing, benchmarking, and CLI surfaces
- `docs/`: framework docs, tutorials, and reference material
- `examples/`: runnable example pipelines, integration patterns, and repro
  targets
- `tests/`: framework-level verification and regression coverage

## Change Practices

- Prefer the smallest owning surface that satisfies the request. Keep changes
  local unless shared behavior is required.
- Add or update focused tests when shared behavior changes.
- Keep docs aligned when documented semantics, guidance, or example-facing
  behavior change.
- If shared core behavior changes, update affected examples so the framework
  and its reference integrations stay aligned.

## Backward Compatibility

Backward compatibility matters for documented public APIs once `ml-pipes` is
published as a package.

At the moment, nothing is published yet, so there is no backward-compatibility
guarantee for the current API surface.

Still, when changing a documented or example-facing surface, update the
relevant docs, examples, and tests together so the repository stays coherent.

## Workflow Router

For workflow-specific guidance, start with
`.agents/skills/README.md` and then read the matching `SKILL.md`.
