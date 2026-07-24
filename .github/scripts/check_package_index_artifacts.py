from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
import urllib.error
import urllib.parse
import urllib.request


@dataclass(frozen=True)
class ArtifactRecord:
    filename: str
    sha256: str


@dataclass(frozen=True)
class IndexArtifactCheck:
    version: str
    matching_filenames: tuple[str, ...]
    missing_filenames: tuple[str, ...]

    @property
    def artifacts_missing(self) -> bool:
        return bool(self.missing_filenames)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check staged release artifacts against a package index, prune matching files, "
            "and fail on conflicting duplicates."
        )
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        required=True,
        help="Directory containing the wheel and sdist for one staged distribution check.",
    )
    parser.add_argument(
        "--dist-name",
        required=True,
        help="Canonical distribution name such as ml-pipes-core.",
    )
    parser.add_argument(
        "--index-url-base",
        required=True,
        help="Base package index URL such as https://test.pypi.org or https://pypi.org.",
    )
    parser.add_argument(
        "--github-output",
        type=Path,
        default=None,
        help="Optional path to the GITHUB_OUTPUT file for workflow step outputs.",
    )
    return parser.parse_args()


def _artifact_identity(filename: str) -> tuple[str, str]:
    if filename.endswith(".whl"):
        parts = filename[:-4].split("-")
        if len(parts) not in {5, 6}:
            raise ValueError(f"Unsupported wheel filename: {filename!r}")
        return parts[0], parts[1]

    for suffix in (".tar.gz", ".zip"):
        if filename.endswith(suffix):
            stem = filename[: -len(suffix)]
            dist_token, version = stem.rsplit("-", 1)
            return dist_token, version

    raise ValueError(f"Unsupported artifact filename: {filename!r}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _local_artifacts(artifacts_dir: Path, expected_dist_name: str) -> tuple[str, dict[str, ArtifactRecord]]:
    artifacts = sorted(path for path in artifacts_dir.iterdir() if path.is_file())
    if not artifacts:
        raise ValueError(f"No artifacts found in {artifacts_dir}")

    expected_dist_token = expected_dist_name.replace("-", "_")
    version: str | None = None
    records: dict[str, ArtifactRecord] = {}
    for artifact in artifacts:
        dist_token, artifact_version = _artifact_identity(artifact.name)
        if dist_token != expected_dist_token:
            raise ValueError(
                f"Artifact {artifact.name!r} does not match expected distribution {expected_dist_name!r}"
            )
        if version is None:
            version = artifact_version
        elif artifact_version != version:
            raise ValueError(
                f"Artifacts in {artifacts_dir} do not share one version: {version!r} and {artifact_version!r}"
            )
        records[artifact.name] = ArtifactRecord(filename=artifact.name, sha256=_sha256(artifact))

    assert version is not None
    return version, records


def _fetch_existing_artifacts(
    index_url_base: str,
    dist_name: str,
    version: str,
) -> dict[str, ArtifactRecord] | None:
    base = index_url_base.rstrip("/")
    url = (
        f"{base}/pypi/{urllib.parse.quote(dist_name, safe='')}/"
        f"{urllib.parse.quote(version, safe='')}/json"
    )
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(request) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise RuntimeError(f"Failed to query {url}: HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Failed to query {url}: {exc.reason}") from exc

    urls = payload.get("urls")
    if not isinstance(urls, list):
        raise RuntimeError(f"Unexpected JSON payload from {url}: missing urls list")

    records: dict[str, ArtifactRecord] = {}
    for file_record in urls:
        if not isinstance(file_record, dict):
            raise RuntimeError(f"Unexpected JSON payload from {url}: invalid file record")
        filename = file_record.get("filename")
        digests = file_record.get("digests")
        if not isinstance(filename, str) or not isinstance(digests, dict):
            raise RuntimeError(f"Unexpected JSON payload from {url}: invalid file metadata")
        sha256 = digests.get("sha256")
        if not isinstance(sha256, str) or not sha256:
            raise RuntimeError(f"Unexpected JSON payload from {url}: missing sha256 digest")
        records[filename] = ArtifactRecord(filename=filename, sha256=sha256)

    return records


def check_package_index_artifacts(
    artifacts_dir: Path,
    *,
    dist_name: str,
    index_url_base: str,
) -> IndexArtifactCheck:
    version, local_artifacts = _local_artifacts(artifacts_dir, dist_name)
    existing_artifacts = _fetch_existing_artifacts(index_url_base, dist_name, version)
    if existing_artifacts is None:
        return IndexArtifactCheck(
            version=version,
            matching_filenames=(),
            missing_filenames=tuple(sorted(local_artifacts)),
        )

    unexpected_existing = sorted(set(existing_artifacts) - set(local_artifacts))
    if unexpected_existing:
        formatted = ", ".join(unexpected_existing)
        raise RuntimeError(
            f"{dist_name} {version} already exists on {index_url_base} with unexpected artifacts: {formatted}"
        )

    matching_filenames: list[str] = []
    missing_filenames: list[str] = []
    conflicts: list[str] = []
    for filename in sorted(local_artifacts):
        local_artifact = local_artifacts[filename]
        existing_artifact = existing_artifacts.get(filename)
        if existing_artifact is None:
            missing_filenames.append(filename)
            continue
        if existing_artifact.sha256 != local_artifact.sha256:
            conflicts.append(filename)
            continue
        matching_filenames.append(filename)

    if conflicts:
        formatted = ", ".join(conflicts)
        raise RuntimeError(
            f"{dist_name} {version} already exists on {index_url_base} with conflicting artifacts: {formatted}"
        )

    for filename in matching_filenames:
        (artifacts_dir / filename).unlink()

    return IndexArtifactCheck(
        version=version,
        matching_filenames=tuple(matching_filenames),
        missing_filenames=tuple(missing_filenames),
    )


def _write_github_output(output_path: Path, artifact_check: IndexArtifactCheck) -> None:
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(f"artifacts_missing={'true' if artifact_check.artifacts_missing else 'false'}\n")
        handle.write(f"version={artifact_check.version}\n")
        handle.write(f"matching_filenames={','.join(artifact_check.matching_filenames)}\n")
        handle.write(f"missing_filenames={','.join(artifact_check.missing_filenames)}\n")


def main() -> int:
    args = _parse_args()
    artifact_check = check_package_index_artifacts(
        args.artifacts_dir,
        dist_name=args.dist_name,
        index_url_base=args.index_url_base,
    )
    if args.github_output is not None:
        _write_github_output(args.github_output, artifact_check)

    if artifact_check.artifacts_missing:
        missing = ", ".join(artifact_check.missing_filenames)
        if artifact_check.matching_filenames:
            matching = ", ".join(artifact_check.matching_filenames)
            print(
                f"{args.dist_name} {artifact_check.version}: missing from index: {missing}; "
                f"already present with matching hashes: {matching}",
                flush=True,
            )
        else:
            print(
                f"{args.dist_name} {artifact_check.version}: missing from index: {missing}",
                flush=True,
            )
    else:
        matching = ", ".join(artifact_check.matching_filenames)
        print(
            f"{args.dist_name} {artifact_check.version}: all artifacts already exist on the index "
            f"with matching hashes ({matching})",
            flush=True,
        )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1) from None
