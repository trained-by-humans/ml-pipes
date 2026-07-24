from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


def _load_release_artifact_manifest_module():
    module_path = (
        Path(__file__).resolve().parents[3] / ".github" / "scripts" / "release_artifact_manifest.py"
    )
    spec = importlib.util.spec_from_file_location("release_artifact_manifest", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_artifact(path: Path, content: str) -> None:
    path.write_bytes(content.encode("utf-8"))


def _release_directory(tmp_path: Path) -> Path:
    artifacts_dir = tmp_path / "release"
    artifacts_dir.mkdir()
    _write_artifact(artifacts_dir / "ml_pipes_core-0.2.0-py3-none-any.whl", "core-wheel")
    _write_artifact(artifacts_dir / "ml_pipes_core-0.2.0.tar.gz", "core-sdist")
    return artifacts_dir


def test_write_release_artifact_manifest_records_expected_artifacts(tmp_path: Path) -> None:
    module = _load_release_artifact_manifest_module()
    artifacts_dir = _release_directory(tmp_path)
    manifest_path = artifacts_dir / module.ARTIFACT_MANIFEST_FILENAME

    records = module.write_release_artifact_manifest(
        artifacts_dir,
        manifest_path,
        tag="v0.2.0rc1",
    )

    assert [record.filename for record in records] == [
        "ml_pipes_core-0.2.0-py3-none-any.whl",
        "ml_pipes_core-0.2.0.tar.gz",
    ]
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["tag"] == "v0.2.0rc1"
    assert [record["filename"] for record in payload["artifacts"]] == [
        "ml_pipes_core-0.2.0-py3-none-any.whl",
        "ml_pipes_core-0.2.0.tar.gz",
    ]


def test_verify_release_artifact_manifest_accepts_matching_artifacts(tmp_path: Path) -> None:
    module = _load_release_artifact_manifest_module()
    artifacts_dir = _release_directory(tmp_path)
    manifest_path = artifacts_dir / module.ARTIFACT_MANIFEST_FILENAME
    module.write_release_artifact_manifest(artifacts_dir, manifest_path, tag="v0.2.0rc1")

    records = module.verify_release_artifact_manifest(
        artifacts_dir,
        manifest_path,
        expected_tag="v0.2.0rc1",
    )

    assert [record.filename for record in records] == [
        "ml_pipes_core-0.2.0-py3-none-any.whl",
        "ml_pipes_core-0.2.0.tar.gz",
    ]


def test_verify_release_artifact_manifest_rejects_unexpected_artifacts(tmp_path: Path) -> None:
    module = _load_release_artifact_manifest_module()
    artifacts_dir = _release_directory(tmp_path)
    manifest_path = artifacts_dir / module.ARTIFACT_MANIFEST_FILENAME
    module.write_release_artifact_manifest(artifacts_dir, manifest_path, tag="v0.2.0rc1")
    _write_artifact(artifacts_dir / "ml_pipes_tensor-0.2.0-py3-none-any.whl", "tensor-wheel")

    with pytest.raises(ValueError, match="Unexpected release artifacts"):
        module.verify_release_artifact_manifest(
            artifacts_dir,
            manifest_path,
            expected_tag="v0.2.0rc1",
        )


def test_verify_release_artifact_manifest_rejects_hash_mismatch(tmp_path: Path) -> None:
    module = _load_release_artifact_manifest_module()
    artifacts_dir = _release_directory(tmp_path)
    manifest_path = artifacts_dir / module.ARTIFACT_MANIFEST_FILENAME
    module.write_release_artifact_manifest(artifacts_dir, manifest_path, tag="v0.2.0rc1")
    _write_artifact(artifacts_dir / "ml_pipes_core-0.2.0.tar.gz", "changed")

    with pytest.raises(ValueError, match="did not match the manifest"):
        module.verify_release_artifact_manifest(
            artifacts_dir,
            manifest_path,
            expected_tag="v0.2.0rc1",
        )
