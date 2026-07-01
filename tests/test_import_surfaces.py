from __future__ import annotations

import subprocess
import sys


def _run_python(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )


def test_ml_pipes_core_import_does_not_eagerly_import_cv2() -> None:
    result = _run_python(
        "import sys; import ml_pipes.core; print('cv2' in sys.modules)"
    )

    assert result.stdout.strip() == "False"


def test_root_pipeline_export_is_lazy_and_does_not_eagerly_import_cv2() -> None:
    result = _run_python(
        "import sys; from ml_pipes import Pipeline; del Pipeline; print('cv2' in sys.modules)"
    )

    assert result.stdout.strip() == "False"


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
