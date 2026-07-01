# Humans

This file keeps the repository-level workflow notes that are useful for human
maintainers but not needed for normal package usage or for agents that can
infer the same information from the workspace metadata and scripts.

## Workspace Development

From a repository checkout, install the workspace members you want to edit:

```bash
python -m pip install \
  -e packages/core \
  -e packages/tensor \
  -e packages/vision \
  -e packages/onnx \
  -e packages/torch \
  -e packages/meta
```

## Release Order

The package dependency graph requires this publish order:

1. `ml-pipes-core`
2. `ml-pipes-tensor`
3. `ml-pipes-vision`
4. `ml-pipes-onnx`
5. `ml-pipes-torch`
6. `ml-pipes`

Run a release dry-run from the repository root with:

```bash
python scripts/release_packages.py --dry-run
```
