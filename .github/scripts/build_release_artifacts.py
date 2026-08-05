from __future__ import annotations

import argparse
import importlib.util
import shlex
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import release_plan as release_plan_module
import validate_release_metadata as validate_release_metadata_module


ROOT = release_plan_module.ROOT
RELEASE_REQUIREMENTS_FILE = ROOT / ".github" / "requirements-release.txt"
validate_release_metadata = validate_release_metadata_module.validate_release_metadata


def _artifact_glob(dist_name: str) -> str:
    return dist_name.replace("-", "_")


def _pip_install_command(*args: str) -> str:
    return shlex.join([sys.executable, "-m", "pip", "install", *args])


def _ensure_build_tooling() -> None:
    required_modules = ["build", "hatchling"]
    missing_modules = [name for name in required_modules if importlib.util.find_spec(name) is None]
    if not missing_modules:
        return

    missing = ", ".join(missing_modules)
    raise RuntimeError(
        "release artifact building requires release tooling in this interpreter. "
        f"Missing modules: {missing}. "
        f"Install them with: {_pip_install_command('-r', str(RELEASE_REQUIREMENTS_FILE))}"
    )


def _build_package(package_dir: Path, outdir: Path, *, dist_name: str) -> None:
    normalized = _artifact_glob(dist_name)
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the published ml-pipes release artifacts in release order."
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=ROOT / "dist" / "release",
        help="Directory where built wheels and sdists are written.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    _version, manifests = validate_release_metadata()
    _ensure_build_tooling()

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, list[Path]] = {}
    for package_manifest in manifests:
        package_dir = ROOT / "packages" / package_manifest.package_dir_name
        print(f"== building {package_manifest.dist_name} ==", flush=True)
        _build_package(package_dir, outdir, dist_name=package_manifest.dist_name)
        manifest[package_manifest.dist_name] = _artifacts_for(package_manifest.dist_name, outdir)

    print("\nBuild order:", flush=True)
    for package_manifest in manifests:
        artifact_names = ", ".join(path.name for path in manifest[package_manifest.dist_name])
        print(f"- {package_manifest.dist_name}: {artifact_names}", flush=True)

    print("\nRelease artifact build complete.", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1) from None
    except subprocess.CalledProcessError as exc:
        command = shlex.join(str(arg) for arg in exc.cmd)
        print(
            f"Error: command failed with exit code {exc.returncode}: {command}",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(exc.returncode) from None
