from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
ZERO_GIT_SHA = "0" * 40

ChangedFilesResolver = Callable[[str | None, str | None, Path], tuple[str, ...]]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Detect which workspace packages changed between two Git revisions and emit them as JSON."
        )
    )
    parser.add_argument(
        "--event-name",
        required=True,
        help="GitHub event name such as pull_request, push, or workflow_call.",
    )
    parser.add_argument(
        "--base-sha",
        default=None,
        help="Base revision for Git diffs. Ignored for workflow_call.",
    )
    parser.add_argument(
        "--head-sha",
        default=None,
        help="Head revision for Git diffs. Ignored for workflow_call.",
    )
    parser.add_argument(
        "--github-output",
        type=Path,
        default=None,
        help="Optional path to the GITHUB_OUTPUT file for workflow step outputs.",
    )
    return parser.parse_args()


def _workspace_packages(root: Path = ROOT) -> tuple[str, ...]:
    packages_dir = root / "packages"
    if not packages_dir.is_dir():
        raise ValueError(f"Workspace packages directory does not exist: {packages_dir}")

    return tuple(
        sorted(
            path.name
            for path in packages_dir.iterdir()
            if path.is_dir() and (path / "pyproject.toml").is_file()
        )
    )


def _changed_files_command(base_sha: str | None, head_sha: str | None) -> tuple[str, ...]:
    if not head_sha:
        raise ValueError("head_sha is required to resolve changed files")
    if not base_sha or base_sha == ZERO_GIT_SHA:
        return ("git", "show", "--pretty=", "--name-only", head_sha)
    return ("git", "diff", "--name-only", base_sha, head_sha)


def _git_changed_files(base_sha: str | None, head_sha: str | None, root: Path = ROOT) -> tuple[str, ...]:
    command = _changed_files_command(base_sha, head_sha)
    try:
        result = subprocess.run(
            command,
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        details = exc.stderr.strip() or exc.stdout.strip() or str(exc)
        raise RuntimeError(f"Failed to list changed files: {details}") from exc

    return tuple(line for line in result.stdout.splitlines() if line)


def detect_changed_packages(
    event_name: str,
    *,
    base_sha: str | None = None,
    head_sha: str | None = None,
    root: Path = ROOT,
    resolve_changed_files: ChangedFilesResolver = _git_changed_files,
) -> tuple[str, ...]:
    packages = _workspace_packages(root)

    if event_name == "workflow_call":
        return packages
    if event_name not in {"pull_request", "push"}:
        return packages
    if not head_sha:
        return packages

    changed_files = resolve_changed_files(base_sha, head_sha, root)
    return tuple(
        package
        for package in packages
        if any(path.startswith(f"packages/{package}/") for path in changed_files)
    )


def _changed_packages_json(changed_packages: tuple[str, ...]) -> str:
    return json.dumps(list(changed_packages), separators=(",", ":"))


def _write_github_output(output_path: Path, changed_packages: tuple[str, ...]) -> None:
    payload = _changed_packages_json(changed_packages)
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(f"changed_packages={payload}\n")


def main() -> int:
    args = _parse_args()
    changed_packages = detect_changed_packages(
        args.event_name,
        base_sha=args.base_sha,
        head_sha=args.head_sha,
    )
    if args.github_output is not None:
        _write_github_output(args.github_output, changed_packages)
    print(_changed_packages_json(changed_packages), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1) from None
