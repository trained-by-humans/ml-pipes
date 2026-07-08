# Skills Index

This file is the workflow router for the agent.
Use it to run the high-level loop: turn requests into concrete changes, route
each change to the right target, and then apply the matching skill.

## Agent Roles

This repo assumes two common agent roles:

- `integrator`: uses `ml-pipes` to build, adapt, or debug a concrete pipeline
  in `examples/` or downstream code
- `maintainer`: changes the shared framework, package-owned public surfaces,
  tests, or docs

Do not switch between `integrator` and `maintainer` roles without explicit
user approval:

- If local pipeline work appears to require framework or package changes, ask
  before switching to maintainer work.
- If framework work turns out to be only local example or integration work,
  ask before switching to integrator work.
- Exception: a maintainer may update examples when those changes are required
  to verify or reflect an approved shared change.

## Workflow Loop

Maintainer work moves through three stages:

1. `<Concrete Changes Stage>` turns the full request into one or more
   concrete changes.
2. `<Target Stage>` takes one concrete change at a time and narrows it to one
   or more targeted changes.
3. `<Placement Stage>` takes one targeted change at a time and uses the
   matching skill to confirm and place it.

The stages are not one big batch. Start by making the whole request concrete.
After that, process each change gradually: target one concrete change, split
it further if needed, and place each targeted change before moving on.

Each placement pass should end with a concrete target location: package,
module, file, and line when known.

```text
request
    -> <Concrete Changes Stage>
    -> for each concrete change:
         -> <Target Stage>
         -> for each targeted change:
              -> <Placement Stage>
```

## Concrete Changes Stage

Turn the user request into one or more candidate concrete changes by reading
the relevant docs and code.

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
change before trying to place it with the relevant implementation workflow.

```text
Input: Concrete Change
- Run [maintainer-triage] to propose the target package/location
- Reuse any rejection reasoning from a previous placement attempt
- Make sure the target is narrow enough to place the change into a specific
  package, module, file, and line when known
- If the target varies, split the change into smaller ones with a clear target
- If triage still reports unresolved ambiguity, stop before placement
- Output: Targeted Change(s)
```

## Placement Stage

Once the change is concrete and has a proposed target location, use the
selected maintainer workflow to confirm whether it really belongs there.

```text
Input: Targeted Change
- <Select Skill By Target> to choose the next workflow
- Apply the selected skill guidance to confirm and place the change
- If placement does not succeed, loop back through <Target Stage> with that
  reasoning
```

## Select Skill By Target

Use this step only when the change is already concrete and has a specific
target location.
This step only selects which skill guidance to apply next.
It does not process the change itself.

Order matters:

```text
when(target):
    is examples/**:
        if debugging: [pipeline-debugger]
        else: [pipeline-builder]
    is docs/** or packages/core/**:
        [maintainer-core]
    is packages/{tensor,vision,onnx,torch}/**:
        [maintainer-packages]
    else:
        [maintainer-triage]
```

## Skills

- `pipeline-builder`: compose, adapt, simplify, or document a concrete
  pipeline using the current package surfaces
- `pipeline-debugger`: localize the first bad step or boundary and identify the
  concrete follow-up target
- `maintainer-packages`: change a package-owned shared surface and its
  package docs/exports
- `maintainer-core`: change core-owned runtime or tooling behavior under
  `packages/core/`
- `maintainer-triage`: propose the target location for concrete changes,
  split mixed changes, and identify the target of each one

Choose the skill that best matches the current task.
