from __future__ import annotations

import argparse
from dataclasses import dataclass
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:
    try:
        import tomli as tomllib
    except ModuleNotFoundError as exc:
        tomllib = None
        _TOML_IMPORT_ERROR = exc
    else:
        _TOML_IMPORT_ERROR = None
else:
    _TOML_IMPORT_ERROR = None

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
INTERNAL_DIST_NAMES = {dist_name for _, dist_name in PACKAGE_ORDER}
PACKAGE_ORDER_INDEX = {dist_name: index for index, (_, dist_name) in enumerate(PACKAGE_ORDER)}
_REQUIREMENT_NAME_RE = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")


@dataclass(frozen=True)
class PackageManifest:
    package_dir_name: str
    dist_name: str
    version: str
    runtime_internal_dependencies: tuple[str, ...]


def _artifact_glob(dist_name: str) -> str:
    return dist_name.replace("-", "_")


def _load_pyproject(package_dir: Path) -> dict[str, Any]:
    if tomllib is None:
        raise RuntimeError(
            "release validation requires a TOML parser to read pyproject.toml. "
            "Use Python 3.11+ or install tomli when running under Python 3.10."
        ) from _TOML_IMPORT_ERROR
    with (package_dir / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)


def _requirement_name(requirement: str) -> str:
    match = _REQUIREMENT_NAME_RE.match(requirement)
    if match is None:
        raise ValueError(f"Unsupported dependency requirement format: {requirement!r}")
    return match.group(1)


def _runtime_internal_dependencies(project: dict[str, Any]) -> tuple[str, ...]:
    dependencies = project.get("dependencies", [])
    internal_dependencies: list[str] = []
    for requirement in dependencies:
        name = _requirement_name(requirement)
        if name in INTERNAL_DIST_NAMES:
            internal_dependencies.append(name)
    return tuple(internal_dependencies)


def _package_manifest(package_dir_name: str, expected_dist_name: str) -> PackageManifest:
    package_dir = ROOT / "packages" / package_dir_name
    pyproject = _load_pyproject(package_dir)
    project = pyproject.get("project")
    if not isinstance(project, dict):
        raise ValueError(f"{package_dir / 'pyproject.toml'} is missing a [project] table")

    dist_name = project.get("name")
    if dist_name != expected_dist_name:
        raise ValueError(
            f"{package_dir / 'pyproject.toml'} declares project.name={dist_name!r}, "
            f"expected {expected_dist_name!r}"
        )

    version = project.get("version")
    if not isinstance(version, str) or not version:
        raise ValueError(f"{package_dir / 'pyproject.toml'} is missing project.version")

    return PackageManifest(
        package_dir_name=package_dir_name,
        dist_name=expected_dist_name,
        version=version,
        runtime_internal_dependencies=_runtime_internal_dependencies(project),
    )


def validate_release_metadata(expected_tag: str | None = None) -> tuple[str, list[PackageManifest]]:
    manifests = [_package_manifest(package_dir_name, dist_name) for package_dir_name, dist_name in PACKAGE_ORDER]

    versions = {manifest.version for manifest in manifests}
    if len(versions) != 1:
        formatted_versions = ", ".join(
            f"{manifest.dist_name}={manifest.version}" for manifest in manifests
        )
        raise ValueError(f"All published packages must share one version; found {formatted_versions}")
    version = versions.pop()

    if expected_tag is not None and expected_tag != f"v{version}":
        raise ValueError(
            f"Release tag {expected_tag!r} does not match package version 'v{version}'"
        )

    for manifest in manifests:
        for dependency in manifest.runtime_internal_dependencies:
            if dependency not in PACKAGE_ORDER_INDEX:
                raise ValueError(
                    f"{manifest.dist_name} depends on unknown internal package {dependency!r}"
                )
            if PACKAGE_ORDER_INDEX[dependency] >= PACKAGE_ORDER_INDEX[manifest.dist_name]:
                raise ValueError(
                    f"Publish order is invalid: {manifest.dist_name} depends on {dependency}, "
                    "but the dependency is not published earlier"
                )

    return version, manifests


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
    parser.add_argument(
        "--tag",
        default=None,
        help="Expected release tag such as v0.1.0 when validating release metadata.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate package versions and the explicit publish order without building artifacts.",
    )
    args = parser.parse_args()
    if args.publish and args.dry_run:
        parser.error("--publish and --dry-run are mutually exclusive")
    if args.validate and (args.publish or args.dry_run):
        parser.error("--validate cannot be combined with --publish or --dry-run")
    if args.tag and not args.validate:
        parser.error("--tag requires --validate")
    return args


def main() -> int:
    args = _parse_args()
    if args.validate:
        version, manifests = validate_release_metadata(args.tag)
        print(f"Validated unified version: {version}", flush=True)
        if args.tag is not None:
            print(f"Validated release tag: {args.tag}", flush=True)
        print("\nPublish order:", flush=True)
        for manifest in manifests:
            dependencies = ", ".join(manifest.runtime_internal_dependencies) or "none"
            print(f"- {manifest.dist_name}: runtime deps -> {dependencies}", flush=True)
        return 0

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
