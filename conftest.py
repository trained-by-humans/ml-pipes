from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PACKAGE_SRC_DIRS = [
    ROOT / "packages" / "core"   / "src",
    ROOT / "packages" / "tensor" / "src",
    ROOT / "packages" / "vision" / "src",
    ROOT / "packages" / "onnx"   / "src",
    ROOT / "packages" / "torch"  / "src",
]

for path in reversed(PACKAGE_SRC_DIRS):
    rendered = str(path)
    if rendered not in sys.path:
        sys.path.insert(0, rendered)
