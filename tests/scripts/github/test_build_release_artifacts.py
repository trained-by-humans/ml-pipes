from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


def _load_build_release_artifacts_module():
    module_path = Path(__file__).resolve().parents[3] / ".github" / "scripts" / "build_release_artifacts.py"
    spec = importlib.util.spec_from_file_location("build_release_artifacts", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _fake_manifests() -> list[SimpleNamespace]:
    return [
        SimpleNamespace(package_dir_name="core", dist_name="ml-pipes-core"),
        SimpleNamespace(package_dir_name="tensor", dist_name="ml-pipes-tensor"),
        SimpleNamespace(package_dir_name="meta", dist_name="ml-pipes"),
    ]


def test_ensure_build_tooling_requires_build_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_build_release_artifacts_module()

    def fake_find_spec(name: str) -> object | None:
        if name in {"build", "hatchling"}:
            return None
        return object()

    monkeypatch.setattr(module.importlib.util, "find_spec", fake_find_spec)

    with pytest.raises(RuntimeError, match="Missing modules: build, hatchling"):
        module._ensure_build_tooling()


def test_main_validates_release_metadata_before_building(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_build_release_artifacts_module()

    monkeypatch.setattr(module, "_parse_args", lambda: argparse.Namespace(outdir=tmp_path))

    events: list[tuple[str, object]] = []

    def fake_validate_release_metadata(expected_tag: str | None = None) -> tuple[str, list[SimpleNamespace]]:
        events.append(("validate", expected_tag))
        return ("0.2.0", _fake_manifests())

    def fake_ensure_build_tooling() -> None:
        events.append(("ensure_build_tooling", True))

    def fake_build_package(package_dir: Path, outdir: Path, *, dist_name: str) -> None:
        events.append(("build", package_dir.name, dist_name))

    def fake_artifacts_for(dist_name: str, outdir: Path) -> list[Path]:
        return [outdir / f"{dist_name}.whl"]

    monkeypatch.setattr(module, "validate_release_metadata", fake_validate_release_metadata)
    monkeypatch.setattr(module, "_ensure_build_tooling", fake_ensure_build_tooling)
    monkeypatch.setattr(module, "_build_package", fake_build_package)
    monkeypatch.setattr(module, "_artifacts_for", fake_artifacts_for)

    assert module.main() == 0

    assert events[0] == ("validate", None)
    assert events[1] == ("ensure_build_tooling", True)
    assert [event for event in events if event[0] == "build"] == [
        ("build", manifest.package_dir_name, manifest.dist_name)
        for manifest in _fake_manifests()
    ]
