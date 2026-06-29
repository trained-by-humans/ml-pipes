# Skills Index

Repository documentation Markdown files are the source of truth.
Use `docs/README.md` as the shared doc index.
Use this file only to pick the right workflow and the right docs for the
requested change.

## Change Surfaces

- `ml-pipes` is the framework surface that contains core + operator-packages
  such as `ml-pipes/ops.py` or `ml-pipes/torch/ops.py`.
- `examples/` are reference apps and repro targets.

## Agent Roles

This repo assumes two common agent roles:

- `integrator`: uses `ml-pipes` to build, adapt, or debug pipelines in
  examples or downstream code
- `maintainer`: changes `ml-pipes` itself, including shared runtime behavior,
  operator packages, tests, and framework docs

Do not switch between `integrator` and `maintainer` roles without explicit
user approval:

- If local pipeline work appears to require framework changes, ask before
switching into a maintainer skill.
- If framework work turns out to be only local example or integration work, ask
before switching into an integrator skill.

## Route By Change Locality

Start from the outermost scope that can satisfy the request. Keep the change
local unless the requested behavior clearly belongs in a shared layer.

1. `examples/` scope
   Read `examples/README.md` and the closest runnable example or repro target.
   If the request is about one pipeline, local wiring, model quirks, or
   example docs, stay in `examples/` and use `pipeline-builder`.
   If the first broken step or boundary is still unclear, use
   `pipeline-debugger`.

2. operator-package scope
   Read `docs/OPERATORS.md` and `docs/operators/README.md`.
   If the requested behavior should be shared across more than one pipeline
   and belongs in operator composition rather than runtime semantics, use
   `maintainer-operators`.
   Work in the operator-facing surfaces under `src/ml_pipes/` and the matching
   docs/tests.

3. core framework scope
   Read `docs/DESIGN.md` and `docs/ARCHITECTURE.md`.
   If the request changes shared runtime, validation, typing, tracing,
   benchmarking, CLI behavior, or other framework semantics, use
   `maintainer-core`.
   Work in `src/ml_pipes/`, shared docs, and shared tests.

Use `maintainer-triage` when the requested outcome is understood but it is
still unclear whether the change should stay in `examples/`, move into an
operator package, or belong in core.

## Skills

- `pipeline-builder`: compose, adapt, extend, simplify, or document a
  concrete pipeline in `examples/` or downstream code
- `pipeline-debugger`: localize the first failing step, boundary, or failure
  class in an existing pipeline
- `maintainer-operators`: add, change, or document shared behavior in
  operator packages
- `maintainer-core`: implement shared framework-layer changes in
  `src/ml_pipes/`
- `maintainer-triage`: confirm whether a requested change belongs in
  `examples/`, an operator package, or the core library

Choose the skill that best matches the current task.
If the chosen skill explicitly redirects to another skill, follow that
handoff.
