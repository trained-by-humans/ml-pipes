from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage release artifacts for one distribution into a publish directory."
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        required=True,
        help="Directory containing built release artifacts.",
    )
    parser.add_argument(
        "--staging-dir",
        type=Path,
        required=True,
        help="Empty directory where the selected artifacts should be copied.",
    )
    parser.add_argument(
        "--dist-name",
        required=True,
        help="Canonical distribution name such as ml-pipes-core.",
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


def stage_release_artifacts(
    artifacts_dir: Path,
    *,
    staging_dir: Path,
    dist_name: str,
) -> tuple[str, ...]:
    if not artifacts_dir.is_dir():
        raise ValueError(f"Artifacts directory does not exist: {artifacts_dir}")

    if staging_dir.exists():
        if not staging_dir.is_dir():
            raise ValueError(f"Staging path is not a directory: {staging_dir}")
        if any(staging_dir.iterdir()):
            raise ValueError(f"Staging directory must start empty: {staging_dir}")
    else:
        staging_dir.mkdir(parents=True)

    expected_dist_token = dist_name.replace("-", "_")
    staged_filenames: list[str] = []
    for artifact in sorted(path for path in artifacts_dir.iterdir() if path.is_file()):
        dist_token, _version = _artifact_identity(artifact.name)
        if dist_token != expected_dist_token:
            continue
        shutil.copy2(artifact, staging_dir / artifact.name)
        staged_filenames.append(artifact.name)

    if not staged_filenames:
        raise ValueError(
            f"No artifacts for distribution {dist_name!r} were found in {artifacts_dir}"
        )

    return tuple(staged_filenames)


def main() -> int:
    args = _parse_args()
    staged_filenames = stage_release_artifacts(
        args.artifacts_dir,
        staging_dir=args.staging_dir,
        dist_name=args.dist_name,
    )
    formatted = ", ".join(staged_filenames)
    print(f"Staged {args.dist_name} artifacts: {formatted}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1) from None
