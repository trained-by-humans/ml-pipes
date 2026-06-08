from __future__ import annotations

import ast
from pathlib import Path

import pytest

import ml_pipes


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


def _assert_module_exports(module_path: Path) -> None:
    expected = _public_operator_and_alias_names(module_path)

    missing_attrs = sorted(name for name in expected if not hasattr(ml_pipes, name))
    missing_in_all = sorted(name for name in expected if name not in ml_pipes.__all__)
    invalid_in_all = sorted(name for name in ml_pipes.__all__ if not hasattr(ml_pipes, name))

    assert missing_attrs == []
    assert missing_in_all == []
    assert invalid_in_all == []


def test_ml_pipes_exports_all_public_ops_and_aliases() -> None:
    root = Path(__file__).resolve().parents[1] / "src" / "ml_pipes"
    _assert_module_exports(root / "ops.py")
    _assert_module_exports(root / "data_ops.py")


def test_ml_pipes_alias_exports_preserve_identity() -> None:
    assert ml_pipes.GatherScores is ml_pipes.GatherRows
    assert ml_pipes.BinarizeTensor is ml_pipes.CreateTensorMask
    assert ml_pipes.BinarizeTensorByThreshold is ml_pipes.CreateTensorMaskByThreshold


def test_ml_pipes_exports_description_types() -> None:
    assert hasattr(ml_pipes, "OperatorArgument")
    assert hasattr(ml_pipes, "OperatorDescription")
    assert hasattr(ml_pipes, "PipelineDescription")
    assert "OperatorArgument" in ml_pipes.__all__
    assert "OperatorDescription" in ml_pipes.__all__
    assert "PipelineDescription" in ml_pipes.__all__


def test_ml_pipes_exports_operator_decorator_and_type_alias() -> None:
    assert callable(ml_pipes.Operator)
    assert hasattr(ml_pipes, "OperatorLike")
    assert "Operator" in ml_pipes.__all__
    assert "OperatorLike" in ml_pipes.__all__


def test_ml_pipes_torch_exports_all_public_ops_and_aliases() -> None:
    torch = pytest.importorskip("torch")
    del torch
    ml_pipes_torch = pytest.importorskip("ml_pipes.torch")
    module_path = Path(__file__).resolve().parents[1] / "src" / "ml_pipes" / "torch" / "ops.py"
    expected = _public_operator_and_alias_names(module_path)

    missing_attrs = sorted(name for name in expected if not hasattr(ml_pipes_torch, name))
    missing_in_all = sorted(name for name in expected if name not in ml_pipes_torch.__all__)
    invalid_in_all = sorted(name for name in ml_pipes_torch.__all__ if not hasattr(ml_pipes_torch, name))

    assert missing_attrs == []
    assert missing_in_all == []
    assert invalid_in_all == []


def test_ml_pipes_torch_alias_exports_preserve_identity() -> None:
    pytest.importorskip("torch")
    ml_pipes_torch = pytest.importorskip("ml_pipes.torch")

    assert ml_pipes_torch.TorchGatherScores is ml_pipes_torch.TorchGatherRows
    assert ml_pipes_torch.TorchBinarizeTensor is ml_pipes_torch.TorchCreateTensorMask
    assert ml_pipes_torch.TorchBinarizeTensorByThreshold is ml_pipes_torch.TorchCreateTensorMaskByThreshold
