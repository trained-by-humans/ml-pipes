from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_mypy_pipeline_generics_smoke() -> None:
    pytest.importorskip("mypy")

    search_path = os.pathsep.join((str(ROOT / "src"), str(ROOT / "examples")))
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
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
