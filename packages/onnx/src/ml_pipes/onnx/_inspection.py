from __future__ import annotations

from ml_pipes.inspection._registry import register_output_formatter
from ml_pipes.inspection.views import OutputBlock, TextBlock

from .types import RuntimeOutputs


def _format_runtime_outputs(value: RuntimeOutputs) -> list[OutputBlock]:
    rows = [(name, str(tensor.array.shape)) for name, tensor in zip(value.names, value.tensors)]
    return [TextBlock(type(value).__name__, rows)]


def register_inspection_formatters() -> None:
    register_output_formatter(RuntimeOutputs, _format_runtime_outputs)
