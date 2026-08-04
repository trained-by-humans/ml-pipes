from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
from typing import Iterator
import venv


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import validate_release_metadata as validate_release_metadata_module


PackageManifest = validate_release_metadata_module.PackageManifest
validate_release_metadata = validate_release_metadata_module.validate_release_metadata
UMBRELLA_SMOKE_PROFILE = "all"


@dataclass(frozen=True)
class VerificationTarget:
    label: str
    requirement: str
    venv_dir_name: str
    smoke_code: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Install the published ml-pipes distributions from a package index, "
            "run pip check, and smoke their public import surfaces."
        )
    )
    parser.add_argument(
        "--index-url",
        required=True,
        help="Simple package index URL used for installs, for example https://test.pypi.org/simple/.",
    )
    parser.add_argument(
        "--extra-index-url",
        default=None,
        help="Optional fallback simple index URL such as https://pypi.org/simple/.",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="Optional release tag such as v0.1.0rc1 used to validate the checked out metadata.",
    )
    return parser.parse_args()


def _venv_python(venv_dir: Path) -> Path:
    if sys.platform == "win32":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _create_virtualenv(venv_dir: Path) -> Path:
    venv.EnvBuilder(with_pip=True).create(venv_dir)
    return _venv_python(venv_dir)


def _run(command: list[str], *, dist_name: str) -> None:
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as exc:
        rendered = shlex.join(str(arg) for arg in exc.cmd)
        raise RuntimeError(
            f"{dist_name}: command failed with exit code {exc.returncode}: {rendered}"
        ) from exc


def _install_command(
    python_executable: Path,
    *,
    requirement: str,
    index_url: str,
    extra_index_url: str | None,
) -> list[str]:
    command = [
        str(python_executable),
        "-m",
        "pip",
        "install",
        "--no-cache-dir",
        "--index-url",
        index_url,
    ]
    if extra_index_url:
        command.extend(["--extra-index-url", extra_index_url])
    command.append(requirement)
    return command


@contextmanager
def _target_workspace(
    workspace_dir: Path,
    *,
    target: VerificationTarget,
) -> Iterator[Path]:
    workspace_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=workspace_dir,
        prefix=f"{target.venv_dir_name}-",
    ) as temp_dir:
        yield Path(temp_dir)


def _public_module_name(package_dir_name: str) -> str:
    if package_dir_name == "core":
        return "ml_pipes.core"
    return f"ml_pipes.{package_dir_name}"


def _smoke_code(package_dir_name: str, *, dist_name: str, version: str) -> str:
    if package_dir_name == "meta":
        return (
            "from importlib import import_module, metadata; "
            f"assert metadata.version({dist_name!r}) == {version!r}; "
            "module = import_module('ml_pipes.core'); "
            "print(metadata.version('ml-pipes'), module.__name__)"
        )

    module_name = _public_module_name(package_dir_name)
    return (
        "from importlib import import_module; "
        f"module = import_module({module_name!r}); "
        "print(module.__name__)"
    )


def _all_profile_smoke_code(
    *,
    dist_name: str,
    version: str,
    public_modules: tuple[str, ...],
) -> str:
    modules_repr = repr(public_modules)
    return (
        "from importlib import import_module, metadata; "
        f"assert metadata.version({dist_name!r}) == {version!r}; "
        f"modules = {modules_repr}; "
        "[import_module(name) for name in modules]; "
        "print(','.join(modules))"
    )


def _verification_targets(
    manifest: PackageManifest,
    *,
    version: str,
    all_public_modules: tuple[str, ...],
) -> tuple[VerificationTarget, ...]:
    base_target = VerificationTarget(
        label=manifest.dist_name,
        requirement=f"{manifest.dist_name}=={version}",
        venv_dir_name=manifest.package_dir_name,
        smoke_code=_smoke_code(
            manifest.package_dir_name,
            dist_name=manifest.dist_name,
            version=version,
        ),
    )
    if manifest.package_dir_name != "meta":
        return (base_target,)

    all_profile = f"{manifest.dist_name}[{UMBRELLA_SMOKE_PROFILE}]"
    return (
        base_target,
        VerificationTarget(
            label=all_profile,
            requirement=f"{all_profile}=={version}",
            venv_dir_name=f"{manifest.package_dir_name}-{UMBRELLA_SMOKE_PROFILE}",
            smoke_code=_all_profile_smoke_code(
                dist_name=manifest.dist_name,
                version=version,
                public_modules=all_public_modules,
            ),
        ),
    )


def verify_published_package_installs(
    index_url: str,
    *,
    extra_index_url: str | None,
    tag: str | None,
    workspace_dir: Path,
) -> tuple[str, ...]:
    version, manifests = validate_release_metadata(tag)
    all_public_modules = tuple(
        _public_module_name(manifest.package_dir_name)
        for manifest in manifests
        if manifest.package_dir_name != "meta"
    )

    verified_targets: list[str] = []
    for manifest in manifests:
        for target in _verification_targets(
            manifest,
            version=version,
            all_public_modules=all_public_modules,
        ):
            print(f"== verifying {target.label} ==", flush=True)
            with _target_workspace(workspace_dir, target=target) as venv_dir:
                python_executable = _create_virtualenv(venv_dir)

                _run(
                    [str(python_executable), "-m", "pip", "install", "--upgrade", "pip"],
                    dist_name=target.label,
                )
                _run(
                    _install_command(
                        python_executable,
                        requirement=target.requirement,
                        index_url=index_url,
                        extra_index_url=extra_index_url,
                    ),
                    dist_name=target.label,
                )
                _run([str(python_executable), "-m", "pip", "check"], dist_name=target.label)
                _run(
                    [str(python_executable), "-c", target.smoke_code],
                    dist_name=target.label,
                )
            verified_targets.append(target.label)
            print(f"Verified {target.label} {version}", flush=True)

    return tuple(verified_targets)


def main() -> int:
    args = _parse_args()
    with tempfile.TemporaryDirectory(prefix="ml-pipes-published-smoke-") as temp_dir:
        verified_targets = verify_published_package_installs(
            args.index_url,
            extra_index_url=args.extra_index_url,
            tag=args.tag,
            workspace_dir=Path(temp_dir),
        )
    print(f"Verified published install targets: {', '.join(verified_targets)}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1) from None
