from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import textwrap


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


def _run_python(code: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    search_path = os.pathsep.join(str(path) for path in PACKAGE_SRC_DIRS)
    env["PYTHONPATH"] = (
        search_path if not env.get("PYTHONPATH") else search_path + os.pathsep + env["PYTHONPATH"]
    )
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        cwd=ROOT,
        env=env,
        text=True,
        check=False,
    )


def test_ml_pipes_core_import_does_not_eagerly_import_cv2() -> None:
    result = _run_python(
        "import sys; import ml_pipes.core; print('cv2' in sys.modules)"
    )

    assert result.returncode == 0, result.stderr or result.stdout
    assert result.stdout.strip() == "False"


def test_core_import_does_not_require_inspection_extras() -> None:
    script = textwrap.dedent(
        """
        import importlib.abc
        import sys

        blocked = {"cv2", "ml_pipes.onnx", "ml_pipes.tensor", "ml_pipes.vision"}

        class Blocker(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname in blocked or any(fullname.startswith(name + ".") for name in blocked):
                    raise ModuleNotFoundError(f"No module named {fullname!r}")
                return None

        sys.meta_path.insert(0, Blocker())

        import ml_pipes.core

        unexpected = sorted(
            name
            for name in sys.modules
            if name in {
                "ml_pipes.inspection.formatters",
                "ml_pipes.inspection.renderer",
                "ml_pipes.inspection.plot_renderer",
                "ml_pipes.inspection.inspector",
                "ml_pipes.inspection.views",
            }
            or name in blocked
            or any(name.startswith(blocked_name + ".") for blocked_name in blocked)
        )
        assert not unexpected, unexpected
        """
    )

    result = _run_python(script)

    assert result.returncode == 0, result.stderr or result.stdout


def test_otel_collector_reports_optional_dependency_error() -> None:
    result = _run_python(
        "from ml_pipes.collectors.otel_collector import OtelCollector\n"
        "try:\n"
        "    OtelCollector()\n"
        "except ImportError as exc:\n"
        "    print(str(exc))\n"
        "else:\n"
        "    print('installed')\n"
    )

    assert result.returncode == 0, result.stderr or result.stdout
    message = result.stdout.strip()
    assert message == "installed" or "optional otel extra" in message
