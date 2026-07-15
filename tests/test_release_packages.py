from __future__ import annotations

import argparse
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
                        f"ml-pipes-onnx=={version}",
                        f"ml-pipes-tensor=={version}",
                        f"ml-pipes-vision=={version}",
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


def _patch_pyprojects(
    monkeypatch: pytest.MonkeyPatch,
    module: object,
    *,
    version: str = "0.2.0",
) -> dict[str, dict[str, object]]:
    pyprojects = _fake_release_pyprojects(version=version)

    def fake_load_pyproject(package_dir: Path) -> dict[str, object]:
        return pyprojects[package_dir.name]

    monkeypatch.setattr(module, "_load_pyproject", fake_load_pyproject)
    return pyprojects


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
    assert any(
        dependency.source == "project.optional-dependencies.inspection"
        for dependency in manifests[0].internal_dependency_requirements
    )
    assert any(
        dependency.source == "project.optional-dependencies.all"
        for dependency in manifests[-1].internal_dependency_requirements
    )


def test_validate_release_metadata_rejects_mismatched_tag() -> None:
    module = _load_release_packages_module()
    _skip_without_toml_parser(module)

    with pytest.raises(ValueError, match="Release tag"):
        module.validate_release_metadata("v9.9.9")


def test_validate_release_metadata_rejects_stale_runtime_internal_pin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_release_packages_module()
    pyprojects = _patch_pyprojects(monkeypatch, module)

    pyprojects["vision"]["project"]["dependencies"][0] = "ml-pipes-core==0.1.0"

    with pytest.raises(ValueError, match="project.dependencies requirement"):
        module.validate_release_metadata("v0.2.0")


def test_validate_release_metadata_rejects_stale_optional_internal_pin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_release_packages_module()
    pyprojects = _patch_pyprojects(monkeypatch, module)

    pyprojects["core"]["project"]["optional-dependencies"]["inspection"][0] = "ml-pipes-onnx==0.1.0"

    with pytest.raises(ValueError, match="project.optional-dependencies.inspection requirement"):
        module.validate_release_metadata("v0.2.0")


def test_validate_release_metadata_requires_exact_internal_pins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_release_packages_module()
    pyprojects = _patch_pyprojects(monkeypatch, module)

    pyprojects["tensor"]["project"]["dependencies"][0] = "ml-pipes-core>=0.2.0"

    with pytest.raises(ValueError, match="must pin an exact version with =="):
        module.validate_release_metadata("v0.2.0")


def test_load_pyproject_requires_toml_parser(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_release_packages_module()

    monkeypatch.setattr(module, "tomllib", None)
    monkeypatch.setattr(module, "_TOML_IMPORT_ERROR", ModuleNotFoundError("No module named 'tomli'"))

    with pytest.raises(RuntimeError, match="install tomli"):
        module._load_pyproject(module.ROOT / "packages" / "core")


def test_ensure_release_tooling_requires_build_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_release_packages_module()

    def fake_find_spec(name: str) -> object | None:
        if name in {"build", "hatchling"}:
            return None
        return object()

    monkeypatch.setattr(module.importlib.util, "find_spec", fake_find_spec)

    with pytest.raises(RuntimeError, match="Missing modules: build, hatchling"):
        module._ensure_release_tooling(include_upload=False)


def test_ensure_release_tooling_requires_twine_for_publish(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_release_packages_module()

    def fake_find_spec(name: str) -> object | None:
        if name == "twine":
            return None
        return object()

    monkeypatch.setattr(module.importlib.util, "find_spec", fake_find_spec)

    with pytest.raises(RuntimeError, match="Missing modules: twine"):
        module._ensure_release_tooling(include_upload=True)


@pytest.mark.parametrize("publish", [False, True], ids=["dry-run", "publish"])
def test_main_validates_release_metadata_before_building(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    publish: bool,
) -> None:
    module = _load_release_packages_module()

    args = argparse.Namespace(
        publish=publish,
        dry_run=not publish,
        validate=False,
        outdir=tmp_path,
        repository_url="https://example.invalid/legacy/" if publish else None,
        tag=None,
    )
    monkeypatch.setattr(module, "_parse_args", lambda: args)

    events: list[tuple[str, object]] = []

    def fake_validate_release_metadata(expected_tag: str | None = None) -> tuple[str, list[object]]:
        events.append(("validate", expected_tag))
        return ("0.2.0", [])

    def fake_ensure_release_tooling(*, include_upload: bool) -> None:
        events.append(("ensure_release_tooling", include_upload))

    def fake_build_package(package_dir: Path, outdir: Path) -> None:
        events.append(("build", package_dir.name))

    def fake_artifacts_for(dist_name: str, outdir: Path) -> list[Path]:
        return [outdir / f"{dist_name}.whl"]

    def fake_publish_package(dist_name: str, outdir: Path, repository_url: str | None) -> None:
        events.append(("publish", dist_name, repository_url))

    monkeypatch.setattr(module, "validate_release_metadata", fake_validate_release_metadata)
    monkeypatch.setattr(module, "_ensure_release_tooling", fake_ensure_release_tooling)
    monkeypatch.setattr(module, "_build_package", fake_build_package)
    monkeypatch.setattr(module, "_artifacts_for", fake_artifacts_for)
    monkeypatch.setattr(module, "_publish_package", fake_publish_package)

    assert module.main() == 0

    assert events[0] == ("validate", None)
    assert events[1] == ("ensure_release_tooling", publish)
    assert [event for event in events if event[0] == "build"] == [
        ("build", package_dir_name) for package_dir_name, _ in module.PACKAGE_ORDER
    ]
    if publish:
        assert [event for event in events if event[0] == "publish"] == [
            ("publish", dist_name, "https://example.invalid/legacy/")
            for _, dist_name in module.PACKAGE_ORDER
        ]
    else:
        assert not [event for event in events if event[0] == "publish"]
