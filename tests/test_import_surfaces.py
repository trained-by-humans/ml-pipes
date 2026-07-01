from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
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
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )


def test_ml_pipes_core_import_does_not_eagerly_import_cv2() -> None:
    result = _run_python(
        "import sys; import ml_pipes.core; print('cv2' in sys.modules)"
    )

    assert result.stdout.strip() == "False"


def test_root_namespace_import_is_lightweight_and_has_no_convenience_exports() -> None:
    result = _run_python(
        "import sys; import ml_pipes; print('cv2' in sys.modules); print(hasattr(ml_pipes, 'Pipeline'))"
    )

    assert result.stdout.splitlines() == ["False", "False"]


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

    message = result.stdout.strip()
    assert message == "installed" or "optional otel extra" in message
