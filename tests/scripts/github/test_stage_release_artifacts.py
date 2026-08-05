from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


def _load_stage_release_artifacts_module():
    module_path = Path(__file__).resolve().parents[3] / ".github" / "scripts" / "stage_release_artifacts.py"
    spec = importlib.util.spec_from_file_location("stage_release_artifacts", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_artifact(path: Path, content: str) -> None:
    path.write_bytes(content.encode("utf-8"))


def _stage_release_directory(tmp_path: Path) -> Path:
    artifacts_dir = tmp_path / "release"
    artifacts_dir.mkdir()
    _write_artifact(artifacts_dir / "ml_pipes_core-0.2.0-py3-none-any.whl", "core-wheel")
    _write_artifact(artifacts_dir / "ml_pipes_core-0.2.0.tar.gz", "core-sdist")
    _write_artifact(artifacts_dir / "ml_pipes_tensor-0.2.0-py3-none-any.whl", "tensor-wheel")
    _write_artifact(artifacts_dir / "ml_pipes-0.2.0-py3-none-any.whl", "meta-wheel")
    _write_artifact(artifacts_dir / "ml_pipes-0.2.0.tar.gz", "meta-sdist")
    _write_artifact(artifacts_dir / "release-artifact-manifest.json", "{}")
    return artifacts_dir


def test_stage_release_artifacts_selects_requested_distribution(tmp_path: Path) -> None:
    module = _load_stage_release_artifacts_module()
    artifacts_dir = _stage_release_directory(tmp_path)
    staging_dir = tmp_path / "publish"

    staged_filenames = module.stage_release_artifacts(
        artifacts_dir,
        staging_dir=staging_dir,
        dist_name="ml-pipes-core",
    )

    assert staged_filenames == (
        "ml_pipes_core-0.2.0-py3-none-any.whl",
        "ml_pipes_core-0.2.0.tar.gz",
    )
    assert sorted(path.name for path in staging_dir.iterdir()) == list(staged_filenames)


def test_stage_release_artifacts_handles_meta_without_prefix_collisions(tmp_path: Path) -> None:
    module = _load_stage_release_artifacts_module()
    artifacts_dir = _stage_release_directory(tmp_path)
    staging_dir = tmp_path / "publish"

    staged_filenames = module.stage_release_artifacts(
        artifacts_dir,
        staging_dir=staging_dir,
        dist_name="ml-pipes",
    )

    assert staged_filenames == (
        "ml_pipes-0.2.0-py3-none-any.whl",
        "ml_pipes-0.2.0.tar.gz",
    )
    assert sorted(path.name for path in staging_dir.iterdir()) == list(staged_filenames)


def test_stage_release_artifacts_rejects_missing_distribution(tmp_path: Path) -> None:
    module = _load_stage_release_artifacts_module()
    artifacts_dir = _stage_release_directory(tmp_path)

    with pytest.raises(ValueError, match="No artifacts for distribution"):
        module.stage_release_artifacts(
            artifacts_dir,
            staging_dir=tmp_path / "publish",
            dist_name="ml-pipes-vision",
        )


def test_stage_release_artifacts_ignores_non_package_metadata(tmp_path: Path) -> None:
    module = _load_stage_release_artifacts_module()
    artifacts_dir = _stage_release_directory(tmp_path)
    staging_dir = tmp_path / "publish"

    staged_filenames = module.stage_release_artifacts(
        artifacts_dir,
        staging_dir=staging_dir,
        dist_name="ml-pipes-core",
    )

    assert "release-artifact-manifest.json" not in staged_filenames
    assert sorted(path.name for path in staging_dir.iterdir()) == list(staged_filenames)
