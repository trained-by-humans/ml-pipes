from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTDIR = ROOT / "dist" / "release"
PACKAGE_ORDER = [
    ("core", "ml-pipes-core"),
    ("tensor", "ml-pipes-tensor"),
    ("vision", "ml-pipes-vision"),
    ("onnx", "ml-pipes-onnx"),
    ("torch", "ml-pipes-torch"),
    ("meta", "ml-pipes"),
]


def _artifact_glob(dist_name: str) -> str:
    return dist_name.replace("-", "_")


def _build_package(package_dir: Path, outdir: Path) -> None:
    normalized = _artifact_glob(f"ml-pipes-{package_dir.name}" if package_dir.name != "meta" else "ml-pipes")
    for stale in outdir.glob(f"{normalized}-*"):
        stale.unlink()
    subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--no-isolation",
            "--wheel",
            "--sdist",
            "--outdir",
            str(outdir),
            str(package_dir),
        ],
        check=True,
    )


def _artifacts_for(dist_name: str, outdir: Path) -> list[Path]:
    normalized = _artifact_glob(dist_name)
    artifacts = sorted(outdir.glob(f"{normalized}-*"))
    if not artifacts:
        raise FileNotFoundError(f"No built artifacts found for {dist_name} in {outdir}")
    return artifacts


def _publish_package(dist_name: str, outdir: Path, repository_url: str | None) -> None:
    command = [sys.executable, "-m", "twine", "upload"]
    if repository_url:
        command.extend(["--repository-url", repository_url])
    command.extend(str(path) for path in _artifacts_for(dist_name, outdir))
    subprocess.run(command, check=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and optionally publish the ml-pipes distributions in dependency order.")
    parser.add_argument(
        "--outdir",
        type=Path,
        default=DEFAULT_OUTDIR,
        help="Directory where built wheels and sdists are written.",
    )
    parser.add_argument(
        "--publish",
        action="store_true",
        help="Upload the built artifacts with twine after building them.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build artifacts and print the publish plan without uploading anything.",
    )
    parser.add_argument(
        "--repository-url",
        default=None,
        help="Optional repository URL passed through to twine upload.",
    )
    args = parser.parse_args()
    if args.publish and args.dry_run:
        parser.error("--publish and --dry-run are mutually exclusive")
    return args


def main() -> int:
    args = _parse_args()
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, list[Path]] = {}
    for package_dir_name, dist_name in PACKAGE_ORDER:
        package_dir = ROOT / "packages" / package_dir_name
        print(f"== building {dist_name} ==", flush=True)
        _build_package(package_dir, outdir)
        manifest[dist_name] = _artifacts_for(dist_name, outdir)

    print("\nPublish order:", flush=True)
    for _, dist_name in PACKAGE_ORDER:
        artifact_names = ", ".join(path.name for path in manifest[dist_name])
        print(f"- {dist_name}: {artifact_names}", flush=True)

    if args.publish:
        for _, dist_name in PACKAGE_ORDER:
            print(f"\n== publishing {dist_name} ==", flush=True)
            _publish_package(dist_name, outdir, args.repository_url)
    else:
        print("\nRelease dry-run complete. Artifacts were built but not uploaded.", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
