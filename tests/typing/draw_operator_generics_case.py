from __future__ import annotations

from typing import cast

import numpy as np

try:
    from typing import assert_type
except ImportError:  # pragma: no cover
    from typing_extensions import assert_type

from ml_pipes.tensor import TensorRegistry
from ml_pipes.vision import DrawBoxes, DrawMasks, ImagePayload


source = ImagePayload(array=np.zeros((8, 8, 3), dtype=np.uint8), color_space="BGR", layout="HWC")
registry = cast(TensorRegistry, None)

assert_type(DrawBoxes()(source, registry), tuple[ImagePayload, TensorRegistry])
assert_type(DrawMasks()(source, registry), tuple[ImagePayload, TensorRegistry])
