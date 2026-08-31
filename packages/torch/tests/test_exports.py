from __future__ import annotations

import ast
from pathlib import Path

import pytest


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in (here, *here.parents):
        if (
            (candidate / "pyproject.toml").is_file()
            and (candidate / "packages" / "core" / "src").is_dir()
        ):
            return candidate
    raise RuntimeError("Could not locate repository root")


ROOT = _repo_root()
MODULE_PATHS = [
    ROOT / "packages" / "torch" / "src" / "ml_pipes" / "torch" / "boundary_ops.py",
    ROOT / "packages" / "torch" / "src" / "ml_pipes" / "torch" / "runtime_ops.py",
    ROOT / "packages" / "torch" / "src" / "ml_pipes" / "torch" / "tensor_ops.py",
    ROOT / "packages" / "torch" / "src" / "ml_pipes" / "torch" / "vision_ops.py",
    ROOT / "packages" / "torch" / "src" / "ml_pipes" / "torch" / "types.py",
]


def _public_operator_and_alias_names(module_path: Path) -> set[str]:
    tree = ast.parse(module_path.read_text())
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            names.add(node.name)
            continue
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and not node.targets[0].id.startswith("_")
            and isinstance(node.value, ast.Name)
        ):
            names.add(node.targets[0].id)
    return names


def _assert_module_exports(module: object, module_paths: list[Path]) -> None:
    expected: set[str] = set()
    for module_path in module_paths:
        expected.update(_public_operator_and_alias_names(module_path))

    missing_attrs = sorted(name for name in expected if not hasattr(module, name))
    missing_in_all = sorted(name for name in expected if name not in module.__all__)
    invalid_in_all = sorted(name for name in module.__all__ if not hasattr(module, name))

    assert missing_attrs == []
    assert missing_in_all == []
    assert invalid_in_all == []


def test_ml_pipes_torch_exports_all_public_ops_and_aliases() -> None:
    torch = pytest.importorskip("torch")
    del torch
    ml_pipes_torch = pytest.importorskip("ml_pipes.torch")

    _assert_module_exports(ml_pipes_torch, MODULE_PATHS)


def test_ml_pipes_torch_alias_exports_preserve_identity() -> None:
    pytest.importorskip("torch")
    ml_pipes_torch = pytest.importorskip("ml_pipes.torch")

    assert ml_pipes_torch.GatherScores is ml_pipes_torch.GatherRows
    assert ml_pipes_torch.BinarizeTensor is ml_pipes_torch.CreateTensorMask
    assert ml_pipes_torch.BinarizeTensorByThreshold is ml_pipes_torch.CreateTensorMaskByThreshold


def test_ml_pipes_torch_does_not_export_prefixed_generic_operator_names() -> None:
    pytest.importorskip("torch")
    ml_pipes_torch = pytest.importorskip("ml_pipes.torch")

    removed_names = (
        "TorchAsType",
        "TorchArgMax",
        "TorchApplyTensorMask",
        "TorchBinarizeTensor",
        "TorchBinarizeTensorByThreshold",
        "TorchCollate",
        "TorchCreateTensorMask",
        "TorchCreateTensorMaskByThreshold",
        "TorchFilterTensors",
        "TorchGatherRows",
        "TorchGatherScores",
        "TorchMapTensor",
        "TorchMultiplyTensors",
        "TorchScale",
        "TorchSelectTensors",
        "TorchSigmoid",
        "TorchSqueeze",
        "TorchSlice",
        "TorchSortTensorsBy",
        "TorchSoftmax",
        "TorchTopK",
        "TorchTopKIndices2D",
        "TorchTranspose",
        "TorchSynchronizeTensors",
        "TorchInfer",
        "TorchExtract",
        "TorchDistribute",
        "TorchTensorPayload",
        "TorchRuntimeOutputs",
        "TorchTensorRegistry",
    )

    for name in removed_names:
        assert not hasattr(ml_pipes_torch, name)
        assert name not in ml_pipes_torch.__all__
