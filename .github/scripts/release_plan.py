from __future__ import annotations

from dataclasses import dataclass
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


ROOT = Path(__file__).resolve().parents[2]
RELEASE_PLAN = ROOT / "release-plan.toml"


@dataclass(frozen=True)
class ReleasePackage:
    package_dir_name: str
    dist_name: str


def load_toml(path: Path) -> dict[str, Any]:
    if tomllib is None:
        raise RuntimeError(
            "release tooling requires a TOML parser. "
            "Use Python 3.11+ or install tomli when running under Python 3.10."
        ) from _TOML_IMPORT_ERROR
    with path.open("rb") as handle:
        return tomllib.load(handle)


def load_release_plan(
    plan_path: Path = RELEASE_PLAN,
) -> tuple[ReleasePackage, ...]:
    payload = load_toml(plan_path)
    raw_packages = payload.get("package")
    if not isinstance(raw_packages, list) or not raw_packages:
        raise ValueError(f"Release plan must contain a non-empty package list: {plan_path}")

    packages: list[ReleasePackage] = []
    seen_workspaces: set[str] = set()
    seen_dist_names: set[str] = set()
    for entry in raw_packages:
        if not isinstance(entry, dict):
            raise ValueError(f"Release plan contains an invalid package entry: {plan_path}")

        package_dir_name = entry.get("workspace")
        dist_name = entry.get("dist")
        if not isinstance(package_dir_name, str) or not package_dir_name:
            raise ValueError(f"Release plan workspace must be a non-empty string: {plan_path}")
        if not isinstance(dist_name, str) or not dist_name:
            raise ValueError(f"Release plan dist must be a non-empty string: {plan_path}")
        if package_dir_name in seen_workspaces:
            raise ValueError(f"Release plan contains duplicate workspace entries: {package_dir_name}")
        if dist_name in seen_dist_names:
            raise ValueError(f"Release plan contains duplicate dist entries: {dist_name}")

        packages.append(ReleasePackage(package_dir_name=package_dir_name, dist_name=dist_name))
        seen_workspaces.add(package_dir_name)
        seen_dist_names.add(dist_name)

    return tuple(packages)
