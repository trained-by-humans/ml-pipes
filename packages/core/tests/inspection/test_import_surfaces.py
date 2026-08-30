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
        "blocks = inspector._value_to_blocks(\n"
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


def test_pipeline_inspector_registers_pydantic_formatter_when_pydantic_is_available() -> None:
    result = _run_python(
        "import sys\n"
        "from types import ModuleType\n"
        "pydantic = ModuleType('pydantic')\n"
        "class BaseModel:\n"
        "    pass\n"
        "pydantic.BaseModel = BaseModel\n"
        "sys.modules['pydantic'] = pydantic\n"
        "from ml_pipes.inspection import GroupBlock, PipelineInspector\n"
        "class Response(BaseModel):\n"
        "    model_fields = {'prediction_count': object()}\n"
        "    def __init__(self):\n"
        "        self.prediction_count = 2\n"
        "blocks = PipelineInspector()._value_to_blocks(Response())\n"
        "print(type(blocks[0]).__name__)\n"
        "print(blocks[0].title)\n"
        "print(blocks[0].children[0].rows[0][0])\n"
        "print(blocks[0].children[0].rows[0][1])\n"
    )

    assert result.returncode == 0, result.stderr or result.stdout
    assert result.stdout.strip().splitlines() == [
        "GroupBlock",
        "Response",
        "prediction_count",
        "2",
    ]


def test_pipeline_inspector_registers_pydantic_v1_compatibility_formatter() -> None:
    result = _run_python(
        "import sys\n"
        "from types import ModuleType\n"
        "pydantic = ModuleType('pydantic')\n"
        "pydantic.__path__ = []\n"
        "pydantic_v1 = ModuleType('pydantic.v1')\n"
        "class BaseModel:\n"
        "    pass\n"
        "class PydanticV1BaseModel:\n"
        "    pass\n"
        "pydantic.BaseModel = BaseModel\n"
        "pydantic_v1.BaseModel = PydanticV1BaseModel\n"
        "sys.modules['pydantic'] = pydantic\n"
        "sys.modules['pydantic.v1'] = pydantic_v1\n"
        "from ml_pipes.inspection import GroupBlock, PipelineInspector\n"
        "class Response(PydanticV1BaseModel):\n"
        "    __fields__ = {'prediction_count': object()}\n"
        "    def __init__(self):\n"
        "        self.prediction_count = 2\n"
        "blocks = PipelineInspector()._value_to_blocks(Response())\n"
        "print(type(blocks[0]).__name__)\n"
        "print(blocks[0].title)\n"
        "print(blocks[0].children[0].rows[0][0])\n"
        "print(blocks[0].children[0].rows[0][1])\n"
    )

    assert result.returncode == 0, result.stderr or result.stdout
    assert result.stdout.strip().splitlines() == [
        "GroupBlock",
        "Response",
        "prediction_count",
        "2",
    ]


def test_global_formatter_registration_initializes_builtins_before_overriding_them() -> None:
    result = _run_python(
        "import numpy as np\n"
        "from ml_pipes.inspection import register_step_formatter, register_value_formatter\n"
        "from ml_pipes.inspection._global_registry import global_formatter_registry\n"
        "from ml_pipes.inspection.views import TextBlock\n"
        "from ml_pipes.region import RegionOpener\n"
        "value_formatter = lambda _: [TextBlock('custom', [])]\n"
        "step_formatter = lambda span, image: (None, image)\n"
        "register_value_formatter(np.ndarray, value_formatter, allow_override=True)\n"
        "register_step_formatter(RegionOpener, step_formatter, allow_override=True)\n"
        "registry = global_formatter_registry()\n"
        "print(registry.get_value_formatter(np.ndarray) is value_formatter)\n"
        "print(registry.get_step_formatter(RegionOpener) is step_formatter)\n"
    )

    assert result.returncode == 0, result.stderr or result.stdout
    assert result.stdout.strip().splitlines() == ["True", "True"]


def test_type_only_inspection_exports_import_without_inspection_extra() -> None:
    result = _run_python(
        "import importlib.abc\n"
        "import sys\n"
        "from typing import get_args\n"
        "blocked = {'cv2'}\n"
        "class Blocker(importlib.abc.MetaPathFinder):\n"
        "    def find_spec(self, fullname, path=None, target=None):\n"
        "        if fullname in blocked or any(fullname.startswith(name + '.') for name in blocked):\n"
        "            raise ModuleNotFoundError(f\"No module named {fullname!r}\")\n"
        "        return None\n"
        "sys.meta_path.insert(0, Blocker())\n"
        "from ml_pipes.inspection import Orientation, Renderer\n"
        "print(Renderer.__name__)\n"
        "print(get_args(Orientation))\n"
        "print('cv2' in sys.modules)\n"
    )

    assert result.returncode == 0, result.stderr or result.stdout
    lines = result.stdout.strip().splitlines()
    assert lines[0] == "Renderer"
    assert lines[1] == "('horizontal', 'vertical')"
    assert lines[2] == "False"
