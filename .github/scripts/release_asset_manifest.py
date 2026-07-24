from __future__ import annotations

import argparse
from dataclasses import asdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys


MANIFEST_FILENAME = "release-artifact-manifest.json"
_ARCHIVE_SUFFIXES = (".whl", ".tar.gz", ".zip")


@dataclass(frozen=True)
class ReleaseAssetRecord:
    filename: str
    sha256: str
    size: int


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write or verify a release-asset manifest for built package artifacts."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    write_parser = subparsers.add_parser(
        "write",
        help="Write a manifest for the release artifacts in one directory.",
    )
    write_parser.add_argument(
        "--artifacts-dir",
        type=Path,
        required=True,
        help="Directory containing wheel and sdist artifacts.",
    )
    write_parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Manifest output path, usually inside the artifacts directory.",
    )
    write_parser.add_argument(
        "--tag",
        default=None,
        help="Optional release tag to record in the manifest.",
    )

    verify_parser = subparsers.add_parser(
        "verify",
        help="Verify release artifacts against an existing manifest.",
    )
    verify_parser.add_argument(
        "--artifacts-dir",
        type=Path,
        required=True,
        help="Directory containing downloaded wheel and sdist artifacts.",
    )
    verify_parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Path to a previously written release-asset manifest.",
    )
    verify_parser.add_argument(
        "--tag",
        default=None,
        help="Optional expected release tag to compare against the manifest.",
    )

    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_release_asset(path: Path) -> bool:
    return path.is_file() and any(path.name.endswith(suffix) for suffix in _ARCHIVE_SUFFIXES)


def collect_release_assets(artifacts_dir: Path) -> tuple[ReleaseAssetRecord, ...]:
    if not artifacts_dir.is_dir():
        raise ValueError(f"Artifacts directory does not exist: {artifacts_dir}")

    records = tuple(
        ReleaseAssetRecord(
            filename=artifact.name,
            sha256=_sha256(artifact),
            size=artifact.stat().st_size,
        )
        for artifact in sorted(path for path in artifacts_dir.iterdir() if _is_release_asset(path))
    )
    if not records:
        raise ValueError(f"No release artifacts were found in {artifacts_dir}")
    return records


def write_release_manifest(
    artifacts_dir: Path,
    output_path: Path,
    *,
    tag: str | None = None,
) -> tuple[ReleaseAssetRecord, ...]:
    records = collect_release_assets(artifacts_dir)
    payload: dict[str, object] = {
        "artifacts": [asdict(record) for record in records],
    }
    if tag is not None:
        payload["tag"] = tag

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")

    return records


def _load_release_manifest(manifest_path: Path) -> tuple[str | None, tuple[ReleaseAssetRecord, ...]]:
    with manifest_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, dict):
        raise ValueError(f"Manifest payload must be a JSON object: {manifest_path}")

    raw_tag = payload.get("tag")
    if raw_tag is not None and not isinstance(raw_tag, str):
        raise ValueError(f"Manifest tag must be a string when present: {manifest_path}")

    raw_artifacts = payload.get("artifacts")
    if not isinstance(raw_artifacts, list) or not raw_artifacts:
        raise ValueError(f"Manifest must contain a non-empty artifacts list: {manifest_path}")

    records: list[ReleaseAssetRecord] = []
    seen_filenames: set[str] = set()
    for entry in raw_artifacts:
        if not isinstance(entry, dict):
            raise ValueError(f"Manifest contains an invalid artifact record: {manifest_path}")
        filename = entry.get("filename")
        sha256 = entry.get("sha256")
        size = entry.get("size")
        if not isinstance(filename, str) or not filename:
            raise ValueError(f"Manifest artifact filename must be a non-empty string: {manifest_path}")
        if filename in seen_filenames:
            raise ValueError(f"Manifest contains duplicate artifact filenames: {filename}")
        if not isinstance(sha256, str) or len(sha256) != 64:
            raise ValueError(f"Manifest artifact sha256 must be a 64-character string: {filename}")
        if not isinstance(size, int) or size < 0:
            raise ValueError(f"Manifest artifact size must be a non-negative integer: {filename}")
        records.append(ReleaseAssetRecord(filename=filename, sha256=sha256, size=size))
        seen_filenames.add(filename)

    return raw_tag, tuple(records)


def verify_release_manifest(
    artifacts_dir: Path,
    manifest_path: Path,
    *,
    expected_tag: str | None = None,
) -> tuple[ReleaseAssetRecord, ...]:
    if not artifacts_dir.is_dir():
        raise ValueError(f"Artifacts directory does not exist: {artifacts_dir}")

    manifest_tag, expected_records = _load_release_manifest(manifest_path)
    if expected_tag is not None and manifest_tag != expected_tag:
        raise ValueError(
            f"Manifest tag {manifest_tag!r} does not match expected tag {expected_tag!r}"
        )

    local_artifacts = {
        artifact.name: artifact
        for artifact in sorted(path for path in artifacts_dir.iterdir() if _is_release_asset(path))
    }
    expected_filenames = {record.filename for record in expected_records}
    local_filenames = set(local_artifacts)

    missing = sorted(expected_filenames - local_filenames)
    if missing:
        formatted = ", ".join(missing)
        raise ValueError(f"Manifest expected missing release artifacts: {formatted}")

    unexpected = sorted(local_filenames - expected_filenames)
    if unexpected:
        formatted = ", ".join(unexpected)
        raise ValueError(f"Unexpected release artifacts not tracked by manifest: {formatted}")

    mismatched: list[str] = []
    for record in expected_records:
        artifact = local_artifacts[record.filename]
        if artifact.stat().st_size != record.size or _sha256(artifact) != record.sha256:
            mismatched.append(record.filename)

    if mismatched:
        formatted = ", ".join(mismatched)
        raise ValueError(f"Release artifacts did not match the manifest: {formatted}")

    return expected_records


def main() -> int:
    args = _parse_args()
    if args.command == "write":
        records = write_release_manifest(args.artifacts_dir, args.output, tag=args.tag)
        print(f"Wrote release manifest for {len(records)} artifacts to {args.output}", flush=True)
        return 0
    if args.command == "verify":
        records = verify_release_manifest(args.artifacts_dir, args.manifest, expected_tag=args.tag)
        print(f"Verified release manifest for {len(records)} artifacts from {args.manifest}", flush=True)
        return 0
    raise AssertionError(f"Unsupported command: {args.command!r}")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Error: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1) from None
