from __future__ import annotations

from importlib import import_module
from typing import Any


def load_cv2() -> Any:
    try:
        return import_module("cv2")
    except ImportError as exc:  # pragma: no cover - depends on optional dependency state
        raise ImportError(
            "ml_pipes.inspection requires the optional inspection extra. "
            "Install it with `pip install ml-pipes[inspection]`."
        ) from exc
