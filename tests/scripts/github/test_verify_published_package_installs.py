from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys


def _load_verify_published_package_installs_module():
    module_path = (
        Path(__file__).resolve().parents[3]
        / ".github"
        / "scripts"
        / "verify_published_package_installs.py"
    )
    spec = importlib.util.spec_from_file_location("verify_published_package_installs", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _manifest(module: object, package_dir_name: str, dist_name: str, version: str = "0.2.0"):
    return module.PackageManifest(
        package_dir_name=package_dir_name,
        dist_name=dist_name,
        version=version,
        runtime_internal_dependencies=(),
        internal_dependency_requirements=(),
    )


def test_smoke_code_uses_public_import_surfaces() -> None:
    module = _load_verify_published_package_installs_module()

    core_code = module._smoke_code("core", dist_name="ml-pipes-core", version="0.2.0")
    tensor_code = module._smoke_code("tensor", dist_name="ml-pipes-tensor", version="0.2.0")
    meta_code = module._smoke_code("meta", dist_name="ml-pipes", version="0.2.0")
    all_code = module._all_profile_smoke_code(
        dist_name="ml-pipes",
        version="0.2.0",
        public_modules=("ml_pipes.core", "ml_pipes.tensor"),
    )

    assert "ml_pipes.core" in core_code
    assert "ml_pipes.tensor" in tensor_code
    assert "metadata.version('ml-pipes')" in meta_code
    assert "ml_pipes.tensor" in all_code
    assert "[import_module(name) for name in modules]" in all_code


def test_verification_targets_add_base_and_all_profile_for_meta() -> None:
    module = _load_verify_published_package_installs_module()
    targets = module._verification_targets(
        _manifest(module, "meta", "ml-pipes"),
        version="0.2.0",
        all_public_modules=("ml_pipes.core", "ml_pipes.tensor"),
    )

    assert [target.label for target in targets] == ["ml-pipes", "ml-pipes[all]"]
    assert targets[0].venv_dir_name == "meta"
    assert targets[1].venv_dir_name == "meta-all"
    assert targets[1].requirement == "ml-pipes[all]==0.2.0"


def test_verify_published_package_installs_verifies_release_packages_in_order(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = _load_verify_published_package_installs_module()
    manifests = [
        _manifest(module, "core", "ml-pipes-core"),
        _manifest(module, "tensor", "ml-pipes-tensor"),
        _manifest(module, "meta", "ml-pipes"),
    ]
    commands: list[tuple[str, tuple[str, ...]]] = []
    created_venvs: list[Path] = []

    monkeypatch.setattr(module, "validate_release_metadata", lambda tag=None: ("0.2.0", manifests))

    def fake_create_virtualenv(venv_dir: Path) -> Path:
        created_venvs.append(venv_dir)
        return venv_dir / "bin" / "python"

    def fake_run(command: list[str], *, dist_name: str) -> None:
        commands.append((dist_name, tuple(command)))

    monkeypatch.setattr(module, "_create_virtualenv", fake_create_virtualenv)
    monkeypatch.setattr(module, "_run", fake_run)

    verified_packages = module.verify_published_package_installs(
        "https://test.pypi.org/simple/",
        extra_index_url="https://pypi.org/simple/",
        tag="v0.2.0",
        workspace_dir=tmp_path,
    )

    assert verified_packages == (
        "ml-pipes-core",
        "ml-pipes-tensor",
        "ml-pipes",
        "ml-pipes[all]",
    )
    assert created_venvs == [
        tmp_path / "core",
        tmp_path / "tensor",
        tmp_path / "meta",
        tmp_path / "meta-all",
    ]
    assert commands == [
        (
            "ml-pipes-core",
            (
                str(tmp_path / "core" / "bin" / "python"),
                "-m",
                "pip",
                "install",
                "--upgrade",
                "pip",
            ),
        ),
        (
            "ml-pipes-core",
            (
                str(tmp_path / "core" / "bin" / "python"),
                "-m",
                "pip",
                "install",
                "--index-url",
                "https://test.pypi.org/simple/",
                "--extra-index-url",
                "https://pypi.org/simple/",
                "ml-pipes-core==0.2.0",
            ),
        ),
        (
            "ml-pipes-core",
            (
                str(tmp_path / "core" / "bin" / "python"),
                "-m",
                "pip",
                "check",
            ),
        ),
        (
            "ml-pipes-core",
            (
                str(tmp_path / "core" / "bin" / "python"),
                "-c",
                module._smoke_code("core", dist_name="ml-pipes-core", version="0.2.0"),
            ),
        ),
        (
            "ml-pipes-tensor",
            (
                str(tmp_path / "tensor" / "bin" / "python"),
                "-m",
                "pip",
                "install",
                "--upgrade",
                "pip",
            ),
        ),
        (
            "ml-pipes-tensor",
            (
                str(tmp_path / "tensor" / "bin" / "python"),
                "-m",
                "pip",
                "install",
                "--index-url",
                "https://test.pypi.org/simple/",
                "--extra-index-url",
                "https://pypi.org/simple/",
                "ml-pipes-tensor==0.2.0",
            ),
        ),
        (
            "ml-pipes-tensor",
            (
                str(tmp_path / "tensor" / "bin" / "python"),
                "-m",
                "pip",
                "check",
            ),
        ),
        (
            "ml-pipes-tensor",
            (
                str(tmp_path / "tensor" / "bin" / "python"),
                "-c",
                module._smoke_code("tensor", dist_name="ml-pipes-tensor", version="0.2.0"),
            ),
        ),
        (
            "ml-pipes",
            (
                str(tmp_path / "meta" / "bin" / "python"),
                "-m",
                "pip",
                "install",
                "--upgrade",
                "pip",
            ),
        ),
        (
            "ml-pipes",
            (
                str(tmp_path / "meta" / "bin" / "python"),
                "-m",
                "pip",
                "install",
                "--index-url",
                "https://test.pypi.org/simple/",
                "--extra-index-url",
                "https://pypi.org/simple/",
                "ml-pipes==0.2.0",
            ),
        ),
        (
            "ml-pipes",
            (
                str(tmp_path / "meta" / "bin" / "python"),
                "-m",
                "pip",
                "check",
            ),
        ),
        (
            "ml-pipes",
            (
                str(tmp_path / "meta" / "bin" / "python"),
                "-c",
                module._smoke_code("meta", dist_name="ml-pipes", version="0.2.0"),
            ),
        ),
        (
            "ml-pipes[all]",
            (
                str(tmp_path / "meta-all" / "bin" / "python"),
                "-m",
                "pip",
                "install",
                "--upgrade",
                "pip",
            ),
        ),
        (
            "ml-pipes[all]",
            (
                str(tmp_path / "meta-all" / "bin" / "python"),
                "-m",
                "pip",
                "install",
                "--index-url",
                "https://test.pypi.org/simple/",
                "--extra-index-url",
                "https://pypi.org/simple/",
                "ml-pipes[all]==0.2.0",
            ),
        ),
        (
            "ml-pipes[all]",
            (
                str(tmp_path / "meta-all" / "bin" / "python"),
                "-m",
                "pip",
                "check",
            ),
        ),
        (
            "ml-pipes[all]",
            (
                str(tmp_path / "meta-all" / "bin" / "python"),
                "-c",
                module._all_profile_smoke_code(
                    dist_name="ml-pipes",
                    version="0.2.0",
                    public_modules=("ml_pipes.core", "ml_pipes.tensor"),
                ),
            ),
        ),
    ]


def test_main_passes_arguments_to_verify_published_package_installs(monkeypatch, capsys) -> None:
    module = _load_verify_published_package_installs_module()
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        module,
        "_parse_args",
        lambda: argparse.Namespace(
            index_url="https://pypi.org/simple/",
            extra_index_url=None,
            tag="v0.2.0",
        ),
    )

    def fake_verify_published_package_installs(
        index_url: str,
        *,
        extra_index_url: str | None,
        tag: str | None,
        workspace_dir: Path,
    ) -> tuple[str, ...]:
        captured["index_url"] = index_url
        captured["extra_index_url"] = extra_index_url
        captured["tag"] = tag
        captured["workspace_dir"] = workspace_dir
        return ("ml-pipes-core", "ml-pipes")

    monkeypatch.setattr(
        module,
        "verify_published_package_installs",
        fake_verify_published_package_installs,
    )

    assert module.main() == 0
    assert captured["index_url"] == "https://pypi.org/simple/"
    assert captured["extra_index_url"] is None
    assert captured["tag"] == "v0.2.0"
    assert isinstance(captured["workspace_dir"], Path)
    assert "Verified published install targets: ml-pipes-core, ml-pipes" in capsys.readouterr().out
