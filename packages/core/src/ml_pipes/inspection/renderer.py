from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast

from ml_pipes.inspection.views import StepView

__all__ = [
    "HtmlRenderer",
    "Orientation",
    "Renderer",
]

Orientation = Literal["horizontal", "vertical"]
_ORIENTATIONS: tuple[Orientation, ...] = ("horizontal", "vertical")


class Renderer(Protocol):
    """Anything that can turn a list of StepViews into an output format."""

    def render(
        self,
        views: list[StepView],
        orientation: Orientation = "horizontal",
    ) -> Any: ...


def _normalize_orientation(orientation: str) -> Orientation:
    normalized = orientation.strip().lower()
    if normalized not in _ORIENTATIONS:
        raise ValueError(
            f"Invalid inspection orientation: {orientation!r}. "
            f"Expected one of {list(_ORIENTATIONS)}."
        )
    return cast(Orientation, normalized)


if TYPE_CHECKING:
    from ml_pipes.inspection.html_renderer import HtmlRenderer


def __getattr__(name: str) -> Any:
    if name != "HtmlRenderer":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    value = getattr(import_module(".html_renderer", __package__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
