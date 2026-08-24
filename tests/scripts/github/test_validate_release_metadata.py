from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


def _load_validate_release_metadata_module():
    module_path = Path(__file__).resolve().parents[3] / ".github" / "scripts" / "validate_release_metadata.py"
    spec = importlib.util.spec_from_file_location("validate_release_metadata", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _fake_release_pyprojects(version: str = "0.2.0") -> dict[str, dict[str, object]]:
    return {
        "core": {
            "project": {
                "name": "ml-pipes-core",
                "version": version,
                "dependencies": [
                    "numpy>=1.26",
                ],
                "optional-dependencies": {
                    "inspection": [
                        "opencv-python>=4.9",
                    ],
                },
            },
        },
        "tensor": {
            "project": {
                "name": "ml-pipes-tensor",
                "version": version,
                "dependencies": [
                    f"ml-pipes-core=={version}",
                    "numpy>=1.26",
                ],
            },
        },
        "vision": {
            "project": {
                "name": "ml-pipes-vision",
                "version": version,
                "dependencies": [
                    f"ml-pipes-core=={version}",
                    f"ml-pipes-tensor=={version}",
                    "numpy>=1.26",
                ],
            },
        },
        "onnx": {
            "project": {
                "name": "ml-pipes-onnx",
                "version": version,
                "dependencies": [
                    f"ml-pipes-core=={version}",
                    f"ml-pipes-tensor=={version}",
                    "numpy>=1.26",
                ],
            },
        },
        "torch": {
            "project": {
                "name": "ml-pipes-torch",
                "version": version,
                "dependencies": [
                    f"ml-pipes-core=={version}",
                    f"ml-pipes-tensor=={version}",
                    "numpy>=1.26",
                ],
            },
        },
        "meta": {
            "project": {
                "name": "ml-pipes",
                "version": version,
                "dependencies": [
                    f"ml-pipes-core=={version}",
                ],
                "optional-dependencies": {
                    "all": [
                        f"ml-pipes-core[inspection]=={version}",
                        f"ml-pipes-onnx=={version}",
                        f"ml-pipes-tensor=={version}",
                        f"ml-pipes-torch=={version}",
                        f"ml-pipes-vision=={version}",
                    ],
                },
            },
        },
    }


def _fake_release_packages() -> tuple[SimpleNamespace, ...]:
    return (
        SimpleNamespace(package_dir_name="core", dist_name="ml-pipes-core"),
        SimpleNamespace(package_dir_name="tensor", dist_name="ml-pipes-tensor"),
        SimpleNamespace(package_dir_name="vision", dist_name="ml-pipes-vision"),
        SimpleNamespace(package_dir_name="onnx", dist_name="ml-pipes-onnx"),
        SimpleNamespace(package_dir_name="torch", dist_name="ml-pipes-torch"),
        SimpleNamespace(package_dir_name="meta", dist_name="ml-pipes"),
    )


def _patch_release_inputs(
    monkeypatch: pytest.MonkeyPatch,
    module: object,
    *,
    version: str = "0.2.0",
) -> dict[str, dict[str, object]]:
    pyprojects = _fake_release_pyprojects(version=version)

    def fake_load_pyproject(package_dir: Path) -> dict[str, object]:
        return pyprojects[package_dir.name]

    monkeypatch.setattr(module, "_release_packages", _fake_release_packages)
    monkeypatch.setattr(module, "_load_pyproject", fake_load_pyproject)
    return pyprojects


def test_validate_release_metadata_accepts_valid_manifests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_validate_release_metadata_module()
    _patch_release_inputs(monkeypatch, module)

    version, manifests = module.validate_release_metadata("v0.2.0")

    assert version == "0.2.0"
    assert [manifest.dist_name for manifest in manifests] == [
        package.dist_name for package in _fake_release_packages()
    ]
    assert manifests[0].runtime_internal_dependencies == ()
    assert manifests[1].runtime_internal_dependencies == ("ml-pipes-core",)
    assert manifests[-1].runtime_internal_dependencies == ("ml-pipes-core",)
    assert not any(
        dependency.source == "project.optional-dependencies.inspection"
        and dependency.dist_name in {"ml-pipes-onnx", "ml-pipes-tensor", "ml-pipes-vision"}
        for dependency in manifests[0].internal_dependency_requirements
    )
    assert any(
        dependency.source == "project.optional-dependencies.all"
        for dependency in manifests[-1].internal_dependency_requirements
    )


def test_package_manifest_matches_current_optional_internal_dependency_shape() -> None:
    module = _load_validate_release_metadata_module()

    core_manifest = module._package_manifest("core", "ml-pipes-core")
    meta_manifest = module._package_manifest("meta", "ml-pipes")

    assert not any(
        dependency.source == "project.optional-dependencies.inspection"
        for dependency in core_manifest.internal_dependency_requirements
    )
    assert any(
        dependency.source == "project.optional-dependencies.all"
        for dependency in meta_manifest.internal_dependency_requirements
    )


def test_validate_release_metadata_rejects_mismatched_tag() -> None:
    module = _load_validate_release_metadata_module()

    with pytest.raises(ValueError, match="Release tag"):
        module.validate_release_metadata("v9.9.9")


def test_validate_release_metadata_rejects_stale_runtime_internal_pin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_validate_release_metadata_module()
    pyprojects = _patch_release_inputs(monkeypatch, module)
    pyprojects["vision"]["project"]["dependencies"][0] = "ml-pipes-core==0.1.0"

    with pytest.raises(ValueError, match="project.dependencies requirement"):
        module.validate_release_metadata("v0.2.0")


def test_validate_release_metadata_rejects_stale_optional_internal_pin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_validate_release_metadata_module()
    pyprojects = _patch_release_inputs(monkeypatch, module)
    pyprojects["meta"]["project"]["optional-dependencies"]["all"][0] = "ml-pipes-core[inspection]==0.1.0"

    with pytest.raises(ValueError, match="project.optional-dependencies.all requirement"):
        module.validate_release_metadata("v0.2.0")


def test_validate_release_metadata_allows_optional_internal_publish_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_validate_release_metadata_module()
    pyprojects = _patch_release_inputs(monkeypatch, module)
    pyprojects["core"]["project"]["optional-dependencies"]["inspection"] = [
        "opencv-python>=4.9",
        "ml-pipes-onnx==0.2.0",
        "ml-pipes-tensor==0.2.0",
        "ml-pipes-vision==0.2.0",
    ]

    version, manifests = module.validate_release_metadata("v0.2.0")

    assert version == "0.2.0"
    assert [manifest.dist_name for manifest in manifests] == [
        package.dist_name for package in _fake_release_packages()
    ]


def test_validate_release_metadata_requires_exact_internal_pins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_validate_release_metadata_module()
    pyprojects = _patch_release_inputs(monkeypatch, module)
    pyprojects["tensor"]["project"]["dependencies"][0] = "ml-pipes-core>=0.2.0"

    with pytest.raises(ValueError, match="must pin an exact version with =="):
        module.validate_release_metadata("v0.2.0")


def test_validate_release_metadata_canonicalizes_internal_dependency_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_validate_release_metadata_module()
    pyprojects = _patch_release_inputs(monkeypatch, module)
    pyprojects["tensor"]["project"]["dependencies"][0] = "ml_pipes.core==0.1.0"

    with pytest.raises(ValueError, match="project.dependencies requirement"):
        module.validate_release_metadata("v0.2.0")
