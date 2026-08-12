from __future__ import annotations

from ml_pipes.inspection._registry import register_output_formatter
from ml_pipes.inspection.views import OutputBlock, TextBlock

from .types import TorchTensorRegistry


def _format_torch_tensor_registry(value: TorchTensorRegistry) -> list[OutputBlock]:
    rows = []
    for name, tensor in value._tensors.items():
        rows.append((name, f"{tuple(tensor.shape)}@{tensor.device}"))
    return [TextBlock(type(value).__name__, rows)]


register_output_formatter(TorchTensorRegistry, _format_torch_tensor_registry)
