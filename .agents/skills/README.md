# Skills Index

Use repo-local skills for bounded workflows that need stronger guidance than
the repo-wide rules in `AGENTS.md`.

## Current Skills

- `pipeline-builder`: compose a correct pipeline first, reuse existing
  operators, generate missing local pieces only when composition cannot cover
  the need, and defer optimization until the pipeline already works.
- `pipeline-debugger`: localize why an existing pipeline is broken, validate or
  inspect the failing boundary, reduce the repro, and classify the failure
  before deciding ownership.
- `change-triage`: classify a bug, regression, feature, enhancement, or
  refactor once the requested behavior is understood, decide whether the
  change belongs in the core library, reusable operators, or a local example
  or external pipeline, then verify at that same layer.
- `maintainer-core`: change shared runtime, validation, composition, tracing,
  and benchmarking behavior.
- `maintainer-operators`: add or manage reusable operators and keep operator
  work out of the wrong layer.

## How They Fit Together

- Start with `pipeline-builder` when the task is to create or repair a concrete
  pipeline, prefer existing operators, and keep missing logic local until reuse
  is proven.
- Use `pipeline-debugger` when a pipeline already exists but the broken step,
  failing boundary, or failure class is still unclear.
- Use `change-triage` after debugging when the main question becomes where the
  change belongs.
- Use `maintainer-core` when the fix changes shared runtime semantics in
  `src/ml_pipes/`, shared CLI behavior, or core validation and tracing rules.
- Use `maintainer-operators` when the work is reusable operator logic that
  should not live in core and should not stay local to one pipeline.
- Treat pipeline work as phased:
  first make it correct and validated, then consider tracing or benchmarking.
- Do not start optimization during initial pipeline construction unless
  performance is the blocking issue or the user explicitly asks for
  optimization work.
- Keep optimization inside `pipeline-builder` for now, but as a later phase
  after the pipeline already runs correctly and has a stable repro or baseline.
- Keep ownership decisions inside `change-triage`; keep failure localization
  inside `pipeline-debugger`.

## Directory Rules

- Keep repo-wide rules in `AGENTS.md`.
- Keep each skill in its own folder with a single `SKILL.md` entrypoint.
- Keep `SKILL.md` short and route to existing repo docs instead of duplicating
  them.
- Add `scripts/` only when a workflow needs deterministic execution or the
  same code would otherwise be rewritten repeatedly.
