from __future__ import annotations

import ast
from pathlib import Path

import pytest

import ml_pipes
import ml_pipes.tensor as tensor

ROOT = Path(__file__).resolve().parents[1]


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


def _assert_module_exports(module: object, module_path: Path) -> None:
    expected = _public_operator_and_alias_names(module_path)

    missing_attrs = sorted(name for name in expected if not hasattr(module, name))
    missing_in_all = sorted(name for name in expected if name not in module.__all__)
    invalid_in_all = sorted(name for name in module.__all__ if not hasattr(module, name))

    assert missing_attrs == []
    assert missing_in_all == []
    assert invalid_in_all == []


def test_root_namespace_exposes_no_legacy_convenience_exports() -> None:
    assert not hasattr(ml_pipes, "Pipeline")
    assert not hasattr(ml_pipes, "__all__")


def test_tensor_alias_exports_preserve_identity() -> None:
    assert tensor.GatherScores is tensor.GatherRows
    assert tensor.BinarizeTensor is tensor.CreateTensorMask
    assert tensor.BinarizeTensorByThreshold is tensor.CreateTensorMaskByThreshold


def test_workspace_packages_ship_py_typed_markers() -> None:
    for package_name in ("core", "tensor", "vision", "onnx", "torch"):
        marker = ROOT / "packages" / package_name / "src" / "ml_pipes" / "py.typed"
        assert marker.is_file(), package_name


def test_ml_pipes_torch_exports_all_public_ops_and_aliases() -> None:
    torch = pytest.importorskip("torch")
    del torch
    ml_pipes_torch = pytest.importorskip("ml_pipes.torch")
    module_path = ROOT / "packages" / "torch" / "src" / "ml_pipes" / "torch" / "ops.py"
    _assert_module_exports(ml_pipes_torch, module_path)


def test_ml_pipes_torch_alias_exports_preserve_identity() -> None:
    pytest.importorskip("torch")
    ml_pipes_torch = pytest.importorskip("ml_pipes.torch")

    assert ml_pipes_torch.TorchGatherScores is ml_pipes_torch.TorchGatherRows
    assert ml_pipes_torch.TorchBinarizeTensor is ml_pipes_torch.TorchCreateTensorMask
    assert ml_pipes_torch.TorchBinarizeTensorByThreshold is ml_pipes_torch.TorchCreateTensorMaskByThreshold
