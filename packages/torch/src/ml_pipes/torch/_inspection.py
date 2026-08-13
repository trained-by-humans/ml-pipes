from __future__ import annotations

from ml_pipes.inspection._global_registry import register_value_formatter
from ml_pipes.inspection.views import OutputBlock, TextBlock

from .types import TorchTensorRegistry


def _format_torch_tensor_registry(value: TorchTensorRegistry) -> list[OutputBlock]:
    rows = []
    for name, tensor in value._tensors.items():
        rows.append((name, f"{tuple(tensor.shape)}@{tensor.device}"))
    return [TextBlock(type(value).__name__, rows)]


def register_inspection_formatters() -> None:
    register_value_formatter(TorchTensorRegistry, _format_torch_tensor_registry)
