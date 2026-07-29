from __future__ import annotations

from pathlib import Path


def normalize_direct_example_sys_path(script_file: str | Path, path_entries: list[str]) -> None:
    """Prefer the repo root and drop the example script directory as an import root.

    Root example scripts run as ``python examples/foo.py`` start with ``examples/``
    at the front of ``sys.path``. That can shadow third-party top-level packages
    such as ``torch`` via ``examples/torch``.
    """

    script_dir = Path(script_file).resolve().parent
    project_root = script_dir.parent
    project_root_str = str(project_root)
    script_dir_str = str(script_dir)

    while project_root_str in path_entries:
        path_entries.remove(project_root_str)
    path_entries.insert(0, project_root_str)

    while script_dir_str in path_entries:
        path_entries.remove(script_dir_str)
