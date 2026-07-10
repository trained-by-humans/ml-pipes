from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_workspace_packages_assign_root_py_typed_marker_to_core_only() -> None:
    markers = sorted(
        path.relative_to(ROOT).as_posix()
        for path in ROOT.glob("packages/*/src/ml_pipes/py.typed")
    )

    assert markers == ["packages/core/src/ml_pipes/py.typed"]
