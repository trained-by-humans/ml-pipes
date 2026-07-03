# Skills Index

Use this file only to pick the right workflow and the right owning scope.
For repo-level guidance such as source of truth and repository structure,
follow `AGENTS.md`.
Use `docs/README.md` as the shared doc index and `docs/PACKAGES.md` to
resolve package ownership.

## Agent Roles

This repo assumes two common agent roles:

- `integrator`: uses `ml-pipes` to build, adapt, or debug a concrete pipeline
  in `examples/` or downstream code
- `maintainer`: changes the shared framework, package-owned public surfaces,
  tests, or docs

Do not switch between `integrator` and `maintainer` roles without explicit
user approval:

- If local pipeline work appears to require framework or package changes, ask
  before switching into a maintainer skill.
- If framework work turns out to be only local example or integration work,
  ask before switching into an integrator skill.
- Exception: a maintainer may update examples when those changes are required
  to verify or reflect an approved shared change.

## Workflow Loop

Process maintainer work in three stages:

1. `<Concrete Changes Stage>`
2. `<Target Stage>`
3. `<Placement Stage>`

Each pass should end with a concrete target location: package, module, file,
and line when known.

```text
request
    -> <Concrete Changes Stage>
    -> for each concrete change:
         -> <Target Stage>
         -> <Placement Stage>
```

## Concrete Changes Stage

The main agent turns the user request into one or more candidate concrete
changes by reading the relevant docs and code.

```text
Input: User Request
- Interpret the request into candidate concrete changes
- Keep changes local at first
- If a change has to move into the framework, keep it in memory instead of
  writing it immediately
- Split big requests with multiple change surfaces into smaller changes
- Output: Concrete Change(s)
```

## Target Stage

Use `[maintainer-triage]` to propose a target location for each concrete
change before trying to place it with an implementation skill.

```text
Input: Concrete Change
- Run [maintainer-triage] to propose the target package/location
- Make sure the target is narrow enough to place the change into a specific
  package, module, file, and line when known
- If the target varies, split the change into smaller ones with a clear target
- Output: Targeted Change(s)
```

## Placement Stage

Once the change is concrete and has a proposed target location, let the
maintainer skill for that target confirm whether it really belongs there.

```text
Input: Targeted Change
- <Select Skill By Target> to select the skill
- Run the selected skill
- If the skill accepts the target: place the current change
- If the skill rejects the change (with a suggestion or reasoning): loop back through <Target Stage>
```

## Select Skill By Target

Use this step only when the change is already concrete and has a specific
target location.
This step only selects the skill. It does not process the change itself.

Order matters:

```text
when(target):
    is examples/**:
        if debugging: [pipeline-debugger]
        else: [pipeline-builder]
    is docs/** or packages/core/**:
        [maintainer-core]
    is packages/{tensor,vision,onnx,torch}/**:
        [maintainer-operators]
    else:
        [maintainer-triage]
```

## Skills

- `pipeline-builder`: compose, adapt, simplify, or document a concrete
  pipeline using the current package surfaces
- `pipeline-debugger`: localize the first bad step or boundary and return the
  concrete follow-up target
- `maintainer-operators`: change a package-owned shared surface and its
  package docs/exports
- `maintainer-core`: change core-owned runtime or tooling behavior under
  `packages/core/`
- `maintainer-triage`: propose the target location for concrete changes,
  split mixed changes, and identify the target of each one

Choose the skill that best matches the current task.
If a skill returns corrected target reasoning or a suggested target, loop
back through `## Target Stage` and `## Placement Stage` before continuing.
