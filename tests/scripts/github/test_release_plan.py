from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


def _load_release_plan_module():
    module_path = Path(__file__).resolve().parents[3] / ".github" / "scripts" / "release_plan.py"
    spec = importlib.util.spec_from_file_location("release_plan", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_load_release_plan_reads_current_release_order() -> None:
    module = _load_release_plan_module()

    packages = module.load_release_plan()

    assert [(package.package_dir_name, package.dist_name) for package in packages] == [
        ("core", "ml-pipes-core"),
        ("tensor", "ml-pipes-tensor"),
        ("vision", "ml-pipes-vision"),
        ("onnx", "ml-pipes-onnx"),
        ("torch", "ml-pipes-torch"),
        ("meta", "ml-pipes"),
    ]


def test_load_release_plan_rejects_duplicate_workspace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_release_plan_module()
    monkeypatch.setattr(
        module,
        "load_toml",
        lambda _path: {
            "package": [
                {"workspace": "core", "dist": "ml-pipes-core"},
                {"workspace": "core", "dist": "ml-pipes-tensor"},
            ]
        },
    )

    with pytest.raises(ValueError, match="duplicate workspace"):
        module.load_release_plan()


def test_load_toml_requires_toml_parser(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_release_plan_module()
    monkeypatch.setattr(module, "tomllib", None)
    monkeypatch.setattr(module, "_TOML_IMPORT_ERROR", ModuleNotFoundError("No module named 'tomli'"))

    with pytest.raises(RuntimeError, match="install tomli"):
        module.load_toml(module.RELEASE_PLAN)
