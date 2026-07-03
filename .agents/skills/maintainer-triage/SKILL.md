---
name: maintainer-triage
description: Propose a target location for concrete changes in ml-pipes. Use when Codex needs to check whether a change stays local, lands in a package-owned surface, or belongs in core, split mixed changes by final location, and hand the resulting targets back to the skill router.
---

# Maintainer Triage

Repository documentation Markdown files define semantics. This file only
drives target triage.

Use this skill when the agent already has one or more concrete changes and
the next decision is what target location those changes should use.

## Goal

The goal of this skill is target selection, not interpretation or
implementation.
For mixed requests, split the work into components and target each one
separately.
Start from concrete changes and narrow each one to its intended target
location.
Stop once every component has a concrete target or the narrowest confident
target available.

## Find The Target

1. Start from concrete changes.
   Do not invent the change from scratch here. Begin from the change the
   main agent already inferred from the request, docs, and code.

2. Check the most likely final location for each change.
   Read `examples/README.md`, `docs/PACKAGES.md`, and the relevant package or
   core docs only as needed to identify where the change should end up.

3. Split mixed requests by target.
   If different parts of the request land in different places, separate them
   into different changes rather than keeping one blended task.

4. Narrow the target as far as the evidence supports.
   For each change, identify:
   - package
   - module
   - file
   - line
   If the exact file or line is still unclear, stop at the narrowest
   confident target.

## If Target Is Still Unclear

- Ask whether the change is meant to stay local, land in one package-owned
  surface, or change shared framework behavior.
- If that is still unclear, report the unresolved alternatives and the
  missing information rather than guessing the target.

## Output

The output of this skill should name, for each change:

- the concrete target: package, module, file, and line when known
- the intended final location if the exact file is still unknown
- any unresolved ambiguity that must be settled before routing
