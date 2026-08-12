from __future__ import annotations

import numpy as np

from ml_pipes.inspection._deps import load_cv2
from ml_pipes.inspection._registry import register_output_formatter
from ml_pipes.inspection.views import ImageBlock, OutputBlock, TextBlock, _make_grid

from .types import TensorPayload, TensorRegistry


def _tensor_item_to_heatmap(arr: np.ndarray, layout: str) -> np.ndarray:
    """Render a single (non-batched) tensor item as a false-colour heatmap."""

    cv2 = load_cv2()
    if arr.ndim == 3:
        ch = arr[0] if layout.startswith("C") else arr[:, :, 0]
    else:
        ch = arr
    ch = ch.astype(np.float32)
    mn, mx = ch.min(), ch.max()
    if mx > mn:
        ch = ((ch - mn) / (mx - mn) * 255).astype(np.uint8)
    else:
        ch = np.zeros_like(ch, dtype=np.uint8)
    heat = cv2.applyColorMap(ch, cv2.COLORMAP_VIRIDIS)
    return cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)


def _format_tensor(value: TensorPayload) -> list[OutputBlock]:
    name = type(value).__name__
    arr = value.array
    if arr.ndim == 4 and value.layout.startswith("N"):
        heatmaps = [_tensor_item_to_heatmap(arr[index], value.layout[1:]) for index in range(arr.shape[0])]
        return [ImageBlock(title=f"{name}  {arr.shape}  {value.dtype}", array=_make_grid(heatmaps))]
    return [ImageBlock(title=f"{name}  {arr.shape}  {value.dtype}", array=_tensor_item_to_heatmap(arr, value.layout))]


def _format_tensor_registry(value: TensorRegistry) -> list[OutputBlock]:
    rows = []
    for name, tensor in value._tensors.items():
        shape = str(tuple(tensor.shape)) if hasattr(tensor, "device") else str(tensor.shape)
        device = getattr(tensor, "device", None)
        rows.append((name, f"{shape}@{device}" if device is not None else shape))
    return [TextBlock(type(value).__name__, rows)]


def register_inspection_formatters() -> None:
    register_output_formatter(TensorPayload, _format_tensor)
    register_output_formatter(TensorRegistry, _format_tensor_registry)
