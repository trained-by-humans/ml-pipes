from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys


def _load_detect_changed_packages_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "detect_changed_packages.py"
    spec = importlib.util.spec_from_file_location("detect_changed_packages", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _workspace_root(tmp_path: Path, package_names: tuple[str, ...] = ("core", "tensor", "torch")) -> Path:
    for package_name in package_names:
        package_dir = tmp_path / "packages" / package_name
        package_dir.mkdir(parents=True)
        (package_dir / "pyproject.toml").write_text(
            "[project]\n"
            f'name = "ml-pipes-{package_name}"\n'
            'version = "0.1.0"\n',
            encoding="utf-8",
        )
    return tmp_path


def test_detect_changed_packages_returns_all_workspace_packages_for_workflow_call(tmp_path: Path) -> None:
    module = _load_detect_changed_packages_module()
    root = _workspace_root(tmp_path, ("torch", "core", "meta"))

    changed_packages = module.detect_changed_packages(
        "workflow_call",
        root=root,
        resolve_changed_files=lambda *_args: (_ for _ in ()).throw(AssertionError("unexpected git call")),
    )

    assert changed_packages == ("core", "meta", "torch")


def test_detect_changed_packages_filters_to_changed_package_paths(tmp_path: Path) -> None:
    module = _load_detect_changed_packages_module()
    root = _workspace_root(tmp_path, ("core", "tensor", "torch"))

    changed_packages = module.detect_changed_packages(
        "pull_request",
        base_sha="base",
        head_sha="head",
        root=root,
        resolve_changed_files=lambda *_args: (
            "packages/torch/src/ml_pipes/torch/ops.py",
            "packages/core/README.md",
            "docs/PACKAGES.md",
        ),
    )

    assert changed_packages == ("core", "torch")


def test_detect_changed_packages_returns_empty_tuple_when_no_package_paths_change(tmp_path: Path) -> None:
    module = _load_detect_changed_packages_module()
    root = _workspace_root(tmp_path)

    changed_packages = module.detect_changed_packages(
        "push",
        base_sha="base",
        head_sha="head",
        root=root,
        resolve_changed_files=lambda *_args: ("README.md", ".github/workflows/ci.yml"),
    )

    assert changed_packages == ()


def test_changed_files_command_uses_git_show_when_base_sha_is_missing() -> None:
    module = _load_detect_changed_packages_module()

    assert module._changed_files_command(None, "head") == (
        "git",
        "show",
        "--pretty=",
        "--name-only",
        "head",
    )
    assert module._changed_files_command(module.ZERO_GIT_SHA, "head") == (
        "git",
        "show",
        "--pretty=",
        "--name-only",
        "head",
    )


def test_main_prints_json_and_writes_github_output(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    module = _load_detect_changed_packages_module()
    output_path = tmp_path / "github-output.txt"

    monkeypatch.setattr(
        module,
        "_parse_args",
        lambda: argparse.Namespace(
            event_name="pull_request",
            base_sha="base",
            head_sha="head",
            github_output=output_path,
        ),
    )
    monkeypatch.setattr(module, "detect_changed_packages", lambda *_args, **_kwargs: ("core", "torch"))

    assert module.main() == 0
    assert capsys.readouterr().out.strip() == "[\"core\",\"torch\"]"
    assert output_path.read_text(encoding="utf-8") == "changed_packages=[\"core\",\"torch\"]\n"
