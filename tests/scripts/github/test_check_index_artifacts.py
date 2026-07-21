from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


def _load_check_index_artifacts_module():
    module_path = Path(__file__).resolve().parents[3] / ".github" / "scripts" / "check_index_artifacts.py"
    spec = importlib.util.spec_from_file_location("check_index_artifacts", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_artifact(path: Path, content: str) -> None:
    path.write_bytes(content.encode("utf-8"))


def _stage_core_artifacts(tmp_path: Path) -> Path:
    packages_dir = tmp_path / "publish"
    packages_dir.mkdir()
    _write_artifact(packages_dir / "ml_pipes_core-0.2.0-py3-none-any.whl", "wheel-0.2.0")
    _write_artifact(packages_dir / "ml_pipes_core-0.2.0.tar.gz", "sdist-0.2.0")
    return packages_dir


def test_check_index_artifacts_reports_all_missing_when_version_is_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_check_index_artifacts_module()
    packages_dir = _stage_core_artifacts(tmp_path)

    monkeypatch.setattr(module, "_fetch_existing_artifacts", lambda *_args, **_kwargs: None)

    artifact_check = module.check_index_artifacts(
        packages_dir,
        dist_name="ml-pipes-core",
        index_url_base="https://test.pypi.org",
    )

    assert artifact_check.version == "0.2.0"
    assert artifact_check.matching_filenames == ()
    assert artifact_check.missing_filenames == (
        "ml_pipes_core-0.2.0-py3-none-any.whl",
        "ml_pipes_core-0.2.0.tar.gz",
    )
    assert artifact_check.artifacts_missing is True


def test_check_index_artifacts_prunes_matching_files_when_all_artifacts_match(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_check_index_artifacts_module()
    packages_dir = _stage_core_artifacts(tmp_path)
    _, local_artifacts = module._local_artifacts(packages_dir, "ml-pipes-core")

    monkeypatch.setattr(module, "_fetch_existing_artifacts", lambda *_args, **_kwargs: local_artifacts)

    artifact_check = module.check_index_artifacts(
        packages_dir,
        dist_name="ml-pipes-core",
        index_url_base="https://test.pypi.org",
    )

    assert artifact_check.matching_filenames == (
        "ml_pipes_core-0.2.0-py3-none-any.whl",
        "ml_pipes_core-0.2.0.tar.gz",
    )
    assert artifact_check.missing_filenames == ()
    assert artifact_check.artifacts_missing is False
    assert list(packages_dir.iterdir()) == []


def test_check_index_artifacts_prunes_matching_files_and_reports_missing_ones(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_check_index_artifacts_module()
    packages_dir = _stage_core_artifacts(tmp_path)
    _, local_artifacts = module._local_artifacts(packages_dir, "ml-pipes-core")
    existing = {
        "ml_pipes_core-0.2.0-py3-none-any.whl": local_artifacts["ml_pipes_core-0.2.0-py3-none-any.whl"]
    }

    monkeypatch.setattr(module, "_fetch_existing_artifacts", lambda *_args, **_kwargs: existing)

    artifact_check = module.check_index_artifacts(
        packages_dir,
        dist_name="ml-pipes-core",
        index_url_base="https://test.pypi.org",
    )

    assert artifact_check.matching_filenames == ("ml_pipes_core-0.2.0-py3-none-any.whl",)
    assert artifact_check.missing_filenames == ("ml_pipes_core-0.2.0.tar.gz",)
    assert artifact_check.artifacts_missing is True
    assert sorted(path.name for path in packages_dir.iterdir()) == ["ml_pipes_core-0.2.0.tar.gz"]


def test_check_index_artifacts_rejects_conflicting_duplicate_filenames(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_check_index_artifacts_module()
    packages_dir = _stage_core_artifacts(tmp_path)
    _, local_artifacts = module._local_artifacts(packages_dir, "ml-pipes-core")
    existing = {
        "ml_pipes_core-0.2.0-py3-none-any.whl": module.ArtifactRecord(
            filename="ml_pipes_core-0.2.0-py3-none-any.whl",
            sha256="different",
        ),
        "ml_pipes_core-0.2.0.tar.gz": local_artifacts["ml_pipes_core-0.2.0.tar.gz"],
    }

    monkeypatch.setattr(module, "_fetch_existing_artifacts", lambda *_args, **_kwargs: existing)

    with pytest.raises(RuntimeError, match="conflicting artifacts"):
        module.check_index_artifacts(
            packages_dir,
            dist_name="ml-pipes-core",
            index_url_base="https://test.pypi.org",
        )
    assert sorted(path.name for path in packages_dir.iterdir()) == [
        "ml_pipes_core-0.2.0-py3-none-any.whl",
        "ml_pipes_core-0.2.0.tar.gz",
    ]


def test_check_index_artifacts_rejects_unexpected_existing_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_check_index_artifacts_module()
    packages_dir = _stage_core_artifacts(tmp_path)
    _, local_artifacts = module._local_artifacts(packages_dir, "ml-pipes-core")
    existing = dict(local_artifacts)
    existing["ml_pipes_core-0.2.0-extra.whl"] = module.ArtifactRecord(
        filename="ml_pipes_core-0.2.0-extra.whl",
        sha256="same",
    )

    monkeypatch.setattr(module, "_fetch_existing_artifacts", lambda *_args, **_kwargs: existing)

    with pytest.raises(RuntimeError, match="unexpected artifacts"):
        module.check_index_artifacts(
            packages_dir,
            dist_name="ml-pipes-core",
            index_url_base="https://test.pypi.org",
        )
    assert sorted(path.name for path in packages_dir.iterdir()) == [
        "ml_pipes_core-0.2.0-py3-none-any.whl",
        "ml_pipes_core-0.2.0.tar.gz",
    ]
