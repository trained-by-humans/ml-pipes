from __future__ import annotations

from importlib.machinery import PathFinder
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples._bootstrap import normalize_direct_example_sys_path


def test_normalize_direct_example_sys_path_removes_examples_torch_shadowing() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    script = repo_root / "examples" / "run_yolo8_batch.py"
    path_entries = [str(script.parent)]

    shadowed = PathFinder.find_spec("torch", path_entries)

    assert shadowed is not None
    assert shadowed.origin == str(repo_root / "examples" / "torch" / "__init__.py")

    normalize_direct_example_sys_path(script, path_entries)

    assert path_entries[0] == str(repo_root)
    assert str(script.parent) not in path_entries
    assert PathFinder.find_spec("torch", path_entries) is None
