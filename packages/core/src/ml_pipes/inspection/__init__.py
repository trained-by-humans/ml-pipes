from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

__all__ = [
    "GroupBlock",
    "HtmlRenderer",
    "ImageBlock",
    "InspectionResult",
    "InspectionSerializer",
    "Orientation",
    "OutputBlock",
    "PipelineInspector",
    "Renderer",
    "StepView",
    "TextBlock",
    "ndarray_image_formatter",
]

_LAZY_EXPORTS = {
    "GroupBlock": (".views", "GroupBlock"),
    "HtmlRenderer": (".html_renderer", "HtmlRenderer"),
    "ImageBlock": (".views", "ImageBlock"),
    "InspectionResult": (".artifacts", "InspectionResult"),
    "InspectionSerializer": (".artifacts", "InspectionSerializer"),
    "Orientation": (".renderer", "Orientation"),
    "OutputBlock": (".views", "OutputBlock"),
    "PipelineInspector": (".inspector", "PipelineInspector"),
    "Renderer": (".renderer", "Renderer"),
    "StepView": (".views", "StepView"),
    "TextBlock": (".views", "TextBlock"),
    "ndarray_image_formatter": ("._builtin_formatters", "ndarray_image_formatter"),
}

if TYPE_CHECKING:
    from ml_pipes.inspection.artifacts import InspectionResult, InspectionSerializer
    from ml_pipes.inspection.html_renderer import HtmlRenderer
    from ml_pipes.inspection.renderer import Orientation, Renderer
    from ml_pipes.inspection.inspector import PipelineInspector
    from ml_pipes.inspection._builtin_formatters import ndarray_image_formatter
    from ml_pipes.inspection.views import GroupBlock, ImageBlock, OutputBlock, StepView, TextBlock


def __getattr__(name: str) -> Any:
    export = _LAZY_EXPORTS.get(name)
    if export is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = export
    value = getattr(import_module(module_name, __name__), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
