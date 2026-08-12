from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


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


def test_inspection_result_html_repr_falls_back_without_inspection_extras() -> None:
    result = _run_python(
        "import importlib.abc\n"
        "import sys\n"
        "blocked = {'cv2', 'ml_pipes.onnx', 'ml_pipes.tensor', 'ml_pipes.vision'}\n"
        "class Blocker(importlib.abc.MetaPathFinder):\n"
        "    def find_spec(self, fullname, path=None, target=None):\n"
        "        if fullname in blocked or any(fullname.startswith(name + '.') for name in blocked):\n"
        "            raise ModuleNotFoundError(f\"No module named {fullname!r}\")\n"
        "        return None\n"
        "sys.meta_path.insert(0, Blocker())\n"
        "from ml_pipes.inspection import InspectionResult\n"
        "from ml_pipes.tracing import StepSpan\n"
        "result = InspectionResult([\n"
        "    StepSpan(label='0:Example', start_time=0.0, duration_s=0.01, output_value='ok')\n"
        "])\n"
        "print(result._repr_html_())\n"
    )

    assert result.returncode == 0, result.stderr or result.stdout
    assert result.stdout.strip() == "None"


def test_pipeline_inspector_follows_imported_package_chain_without_vision_or_onnx() -> None:
    result = _run_python(
        "import importlib.abc\n"
        "import sys\n"
        "import numpy as np\n"
        "blocked = {'ml_pipes.onnx', 'ml_pipes.vision'}\n"
        "class Blocker(importlib.abc.MetaPathFinder):\n"
        "    def find_spec(self, fullname, path=None, target=None):\n"
        "        if fullname in blocked or any(fullname.startswith(name + '.') for name in blocked):\n"
        "            raise ModuleNotFoundError(f\"No module named {fullname!r}\")\n"
        "        return None\n"
        "sys.meta_path.insert(0, Blocker())\n"
        "from ml_pipes.inspection import PipelineInspector\n"
        "inspector = PipelineInspector()\n"
        "from ml_pipes.tensor import TensorRegistry\n"
        "blocks = inspector._output_to_blocks(\n"
        "    TensorRegistry({'scores': np.zeros((2, 3), dtype=np.float32)})\n"
        ")\n"
        "print(type(blocks[0]).__name__)\n"
        "print(blocks[0].title)\n"
        "print(blocks[0].rows[0][0])\n"
        "print(blocks[0].rows[0][1])\n"
        "unexpected = sorted(\n"
        "    name\n"
        "    for name in sys.modules\n"
        "    if name in blocked or any(name.startswith(blocked_name + '.') for blocked_name in blocked)\n"
        ")\n"
        "print(unexpected)\n"
    )

    assert result.returncode == 0, result.stderr or result.stdout
    lines = result.stdout.strip().splitlines()
    assert lines[0] == "TextBlock"
    assert lines[1] == "TensorRegistry"
    assert lines[2] == "scores"
    assert "(2, 3)" in lines[3]
    assert lines[4] == "[]"
