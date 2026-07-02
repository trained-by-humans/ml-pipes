from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

__all__ = [
    "GroupBlock",
    "HtmlRenderer",
    "ImageBlock",
    "InspectionResult",
    "InspectionSerializer",
    "OutputBlock",
    "PipelineInspector",
    "PlotRenderer",
    "Renderer",
    "StepView",
    "TextBlock",
]

_LAZY_EXPORTS = {
    "GroupBlock": (".views", "GroupBlock"),
    "HtmlRenderer": (".html_renderer", "HtmlRenderer"),
    "ImageBlock": (".views", "ImageBlock"),
    "InspectionResult": (".artifacts", "InspectionResult"),
    "InspectionSerializer": (".artifacts", "InspectionSerializer"),
    "OutputBlock": (".views", "OutputBlock"),
    "PipelineInspector": (".inspector", "PipelineInspector"),
    "PlotRenderer": (".plot_renderer", "PlotRenderer"),
    "Renderer": (".views", "Renderer"),
    "StepView": (".views", "StepView"),
    "TextBlock": (".views", "TextBlock"),
}

if TYPE_CHECKING:
    from .artifacts import InspectionResult, InspectionSerializer
    from .html_renderer import HtmlRenderer
    from .inspector import PipelineInspector
    from .plot_renderer import PlotRenderer
    from .views import GroupBlock, ImageBlock, OutputBlock, Renderer, StepView, TextBlock


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
