from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in (here, *here.parents):
        if (
            (candidate / "pyproject.toml").is_file()
            and (candidate / "packages" / "core" / "src").is_dir()
        ):
            return candidate
    raise RuntimeError("Could not locate repository root")


ROOT = _repo_root()
PACKAGE_SRC_DIRS = [
    ROOT / "packages" / "core" / "src",
    ROOT / "packages" / "tensor" / "src",
    ROOT / "packages" / "vision" / "src",
    ROOT / "packages" / "onnx" / "src",
    ROOT / "packages" / "torch" / "src",
]


def test_mypy_pipeline_generics_smoke() -> None:
    pytest.importorskip("mypy")

    search_path = os.pathsep.join([*(str(path) for path in PACKAGE_SRC_DIRS), str(ROOT / "examples")])
    env = os.environ.copy()
    env["MYPYPATH"] = (
        search_path if not env.get("MYPYPATH") else search_path + os.pathsep + env["MYPYPATH"]
    )
    env["PYTHONPATH"] = (
        search_path if not env.get("PYTHONPATH") else search_path + os.pathsep + env["PYTHONPATH"]
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "mypy",
            "--config-file",
            os.devnull,
            "--python-version",
            f"{sys.version_info.major}.{sys.version_info.minor}",
            "tests/typing/pipeline_generics_case.py",
            "tests/typing/draw_operator_generics_case.py",
            "tests/typing/map_predictions_to_objects_case.py",
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_mypy_torch_operator_generics_smoke() -> None:
    pytest.importorskip("mypy")
    pytest.importorskip("torch")

    env = os.environ.copy()
    search_path = os.pathsep.join(str(path) for path in PACKAGE_SRC_DIRS)
    env["MYPYPATH"] = search_path
    env["PYTHONPATH"] = search_path

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "mypy",
            "--config-file",
            os.devnull,
            "--python-version",
            f"{sys.version_info.major}.{sys.version_info.minor}",
            "tests/typing/torch_operator_generics_case.py",
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
