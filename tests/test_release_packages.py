from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


def _load_release_packages_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "release_packages.py"
    spec = importlib.util.spec_from_file_location("release_packages", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _skip_without_toml_parser(module: object) -> None:
    if getattr(module, "tomllib", None) is None:
        pytest.skip("release metadata validation requires tomli on Python 3.10 or Python 3.11+")


def test_validate_release_metadata_accepts_current_manifests() -> None:
    module = _load_release_packages_module()
    _skip_without_toml_parser(module)

    version, manifests = module.validate_release_metadata("v0.1.0")

    assert version == "0.1.0"
    assert [manifest.dist_name for manifest in manifests] == [
        dist_name for _, dist_name in module.PACKAGE_ORDER
    ]
    assert manifests[0].runtime_internal_dependencies == ()
    assert manifests[1].runtime_internal_dependencies == ("ml-pipes-core",)
    assert manifests[-1].runtime_internal_dependencies == ("ml-pipes-core",)


def test_validate_release_metadata_rejects_mismatched_tag() -> None:
    module = _load_release_packages_module()
    _skip_without_toml_parser(module)

    with pytest.raises(ValueError, match="Release tag"):
        module.validate_release_metadata("v9.9.9")


def test_load_pyproject_requires_toml_parser(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_release_packages_module()

    monkeypatch.setattr(module, "tomllib", None)
    monkeypatch.setattr(module, "_TOML_IMPORT_ERROR", ModuleNotFoundError("No module named 'tomli'"))

    with pytest.raises(RuntimeError, match="install tomli"):
        module._load_pyproject(module.ROOT / "packages" / "core")
