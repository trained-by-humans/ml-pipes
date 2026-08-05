from __future__ import annotations

import argparse
from dataclasses import dataclass
import re
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import release_plan as release_plan_module


ROOT = release_plan_module.ROOT
load_release_plan = release_plan_module.load_release_plan
load_toml = release_plan_module.load_toml

_REQUIREMENT_NAME_RE = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")
_PINNED_REQUIREMENT_RE = re.compile(
    r"^\s*(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)"
    r"(?:\[[^\]]+\])?\s*==\s*(?P<version>[^;\s]+)\s*(?:;.*)?$"
)
_NORMALIZE_DIST_NAME_RE = re.compile(r"[-_.]+")


@dataclass(frozen=True)
class InternalDependency:
    dist_name: str
    version: str
    source: str
    requirement: str


@dataclass(frozen=True)
class PackageManifest:
    package_dir_name: str
    dist_name: str
    version: str
    runtime_internal_dependencies: tuple[str, ...]
    internal_dependency_requirements: tuple[InternalDependency, ...]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate published package versions, pins, and release order."
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="Expected release tag such as v0.1.0.",
    )
    return parser.parse_args()


def _release_packages() -> tuple[release_plan_module.ReleasePackage, ...]:
    return load_release_plan()


def _internal_dist_names() -> set[str]:
    return {package.dist_name for package in _release_packages()}


def _package_order_index() -> dict[str, int]:
    return {
        package.dist_name: index
        for index, package in enumerate(_release_packages())
    }


def _load_pyproject(package_dir: Path) -> dict[str, Any]:
    return load_toml(package_dir / "pyproject.toml")


def _requirement_name(requirement: str) -> str:
    match = _REQUIREMENT_NAME_RE.match(requirement)
    if match is None:
        raise ValueError(f"Unsupported dependency requirement format: {requirement!r}")
    return match.group(1)


def _normalize_dist_name(name: str) -> str:
    return _NORMALIZE_DIST_NAME_RE.sub("-", name).lower()


def _internal_dependency(requirement: str, *, source: str) -> InternalDependency | None:
    name = _normalize_dist_name(_requirement_name(requirement))
    if name not in _internal_dist_names():
        return None

    match = _PINNED_REQUIREMENT_RE.match(requirement)
    if match is None:
        raise ValueError(
            f"Internal dependency {requirement!r} in {source} must pin an exact version with ==."
        )
    return InternalDependency(
        dist_name=name,
        version=match.group("version"),
        source=source,
        requirement=requirement,
    )


def _internal_dependencies(requirements: Any, *, source: str) -> tuple[InternalDependency, ...]:
    if not isinstance(requirements, list):
        raise ValueError(f"{source} must be a list of dependency strings")

    internal_dependencies: list[InternalDependency] = []
    for requirement in requirements:
        if not isinstance(requirement, str):
            raise ValueError(f"{source} must contain dependency strings")
        dependency = _internal_dependency(requirement, source=source)
        if dependency is not None:
            internal_dependencies.append(dependency)
    return tuple(internal_dependencies)


def _runtime_internal_dependencies(project: dict[str, Any]) -> tuple[InternalDependency, ...]:
    return _internal_dependencies(project.get("dependencies", []), source="project.dependencies")


def _optional_internal_dependencies(project: dict[str, Any]) -> tuple[InternalDependency, ...]:
    optional_dependencies = project.get("optional-dependencies", {})
    if not isinstance(optional_dependencies, dict):
        raise ValueError("project.optional-dependencies must be a table")

    internal_dependencies: list[InternalDependency] = []
    for extra_name, requirements in optional_dependencies.items():
        source = f"project.optional-dependencies.{extra_name}"
        internal_dependencies.extend(_internal_dependencies(requirements, source=source))
    return tuple(internal_dependencies)


def _runtime_internal_dependency_names(
    dependencies: tuple[InternalDependency, ...],
) -> tuple[str, ...]:
    return tuple(dependency.dist_name for dependency in dependencies)


def _validate_internal_dependency_pins(
    manifests: list[PackageManifest],
    *,
    expected_version: str,
) -> None:
    for manifest in manifests:
        for dependency in manifest.internal_dependency_requirements:
            if dependency.version != expected_version:
                raise ValueError(
                    f"{manifest.dist_name} declares {dependency.source} requirement "
                    f"{dependency.requirement!r}, but internal packages must be pinned "
                    f"to =={expected_version}"
                )


def _validate_runtime_publish_order(manifests: list[PackageManifest]) -> None:
    package_order_index = _package_order_index()
    for manifest in manifests:
        for dependency in manifest.runtime_internal_dependencies:
            if package_order_index[dependency] >= package_order_index[manifest.dist_name]:
                raise ValueError(
                    f"Publish order is invalid: {manifest.dist_name} depends on {dependency}, "
                    "but the dependency is not published earlier"
                )


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

    runtime_internal_dependencies = _runtime_internal_dependencies(project)
    optional_internal_dependencies = _optional_internal_dependencies(project)

    return PackageManifest(
        package_dir_name=package_dir_name,
        dist_name=expected_dist_name,
        version=version,
        runtime_internal_dependencies=_runtime_internal_dependency_names(
            runtime_internal_dependencies
        ),
        internal_dependency_requirements=runtime_internal_dependencies + optional_internal_dependencies,
    )


def validate_release_metadata(expected_tag: str | None = None) -> tuple[str, list[PackageManifest]]:
    release_packages = _release_packages()
    manifests = [
        _package_manifest(package.package_dir_name, package.dist_name)
        for package in release_packages
    ]

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

    _validate_internal_dependency_pins(manifests, expected_version=version)
    _validate_runtime_publish_order(manifests)

    return version, manifests


def main() -> int:
    args = _parse_args()
    version, manifests = validate_release_metadata(args.tag)
    print(f"Validated unified version: {version}", flush=True)
    if args.tag is not None:
        print(f"Validated release tag: {args.tag}", flush=True)
    print("\nPublish order:", flush=True)
    for manifest in manifests:
        dependencies = ", ".join(manifest.runtime_internal_dependencies) or "none"
        print(f"- {manifest.dist_name}: runtime deps -> {dependencies}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1) from None
