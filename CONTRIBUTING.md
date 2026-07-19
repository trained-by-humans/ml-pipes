# Contributing

This guide covers local contributor setup and everyday test runs.

## Local Setup

Contributor setup uses the repository `uv` workspace so local package
dependencies resolve to workspace members instead of published packages.

For most contributor work, from the repository root sync the
`shared-framework` group:

```bash
uv sync --group shared-framework
```

If you need the full local workspace, sync `full-workspace`. That group also
adds the umbrella package under `packages/meta/` for checking published
install behavior:

```bash
uv sync --group full-workspace
```

> [!TIP]
> For the full set of available groups, see `[dependency-groups]` in
> [`pyproject.toml`](pyproject.toml).

> [!NOTE]
> These workspace commands are for local contributor development. Once the
> packages are published, consumer installs should use the published package
> names and profiles from the package docs instead.

## Run Tests

From the repository root:

```bash
uv run pytest
```

Match your synced groups to the surfaces you are testing. The
`shared-framework` group covers many shared tests. When you run tests for a
package-specific surface, sync that package group too.

For example, to run a Torch-specific test module:

```bash
uv sync --group torch

uv run pytest packages/torch/tests/test_torch.py
```

Some test surfaces need more than one group. For example, the Torch
inspection tests also need the inspection extras and the local ONNX and
vision packages:

```bash
uv sync --group torch --group inspection-otel

uv run pytest packages/torch/tests/test_inspection.py
```

## CI

GitHub Actions CI covers normal development changes and is reused by the
release workflow as a safety gate for tagged commits.

CI focuses on the shared framework test path, smoke checks for key install
shapes, and verification that the test runs do not leave tracked file
mutations behind.

> [!IMPORTANT]
> Contributors should still run the relevant local tests before pushing.

## Repository Maintenance

### Release Maintenance

Release-specific setup and package publishing workflow live in
[RELEASE.md](RELEASE.md).

GitHub workflow helpers under `.github/scripts/` are workflow-owned
maintenance scripts. Run them locally only when you need to reproduce or
debug CI/release behavior.

### Docs Asset Generation

README and docs-media generation scripts under `scripts/docs_assets/` use a
separate local requirements file instead of the `uv` workspace groups:

```bash
python3 -m pip install -r scripts/docs_assets/requirements.txt
```

If you use the HTML screenshot helper, also install the browser runtime once:

```bash
python3 -m playwright install chromium
```

## IDE Setup

If your editor still shows unresolved imports after `uv sync`, mark these
directories as source roots and refresh indexing:

- `packages/core/src`
- `packages/tensor/src`
- `packages/vision/src`
- `packages/onnx/src`
- `packages/torch/src`
