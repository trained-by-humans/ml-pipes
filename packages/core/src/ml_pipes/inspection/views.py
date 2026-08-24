from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from ml_pipes.inspection._deps import load_cv2
from ml_pipes.tracing import StepSpan


class Summarizable(Protocol):
    def summary(self) -> str: ...


@dataclass
class ImageBlock:
    title: str
    array: np.ndarray
    dim: bool = False
    overlay_array: np.ndarray | None = None

    def summary(self) -> str:
        return self.title


@dataclass
class TextBlock:
    title: str
    rows: list[tuple[str, str]]
    dim: bool = False

    def summary(self) -> str:
        summary = self.title
        rows = self.rows[:3] if self.title == "dict" else self.rows[:1]
        if rows:
            row_summaries = [(f"{key} {value}".rstrip() if key else f"{value}") for key, value in rows]
            if summary:
                summary += "  " + "  |  ".join(row_summaries)
            else:
                summary = "  |  ".join(row_summaries)
        return summary


@dataclass
class GroupBlock:
    title: str
    children: list["OutputBlock"]
    dim: bool = False

    def summary(self) -> str:
        summary = self.title
        child_summaries = [child.summary() for child in self.children[:2]]
        if child_summaries:
            if summary:
                summary += "  " + "  |  ".join(child_summaries)
            else:
                summary = "  |  ".join(child_summaries)
        if len(self.children) > 2:
            suffix = f"+{len(self.children) - 2} more"
            summary = f"{summary}  |  {suffix}" if summary else suffix
        return summary


OutputBlock = ImageBlock | TextBlock | GroupBlock


@dataclass
class StepView:
    label: str
    operator_config: dict[str, Any]
    blocks: list[OutputBlock]
    error: bool = False
    children: list["StepView"] = field(default_factory=list)

    def summary(self) -> str:
        summary = _summarize_blocks(self.blocks)
        if summary:
            return summary
        return "[ERROR]" if self.error else ""


def _summarize_blocks(blocks: list[OutputBlock]) -> str:
    return "  |  ".join(block.summary() for block in blocks)


def _make_grid(images: list[np.ndarray], divider: int = 0) -> np.ndarray:
    """Tile a list of HxWx3 RGB images into a square-ish grid."""

    import math

    n = len(images)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    h, w = images[0].shape[:2]
    gh = rows * h + divider * (rows - 1)
    gw = cols * w + divider * (cols - 1)
    grid = np.full((gh, gw, 3), 180, dtype=np.uint8)
    for idx, img in enumerate(images):
        if img.shape[:2] != (h, w):
            img = load_cv2().resize(img, (w, h))
        r, c = divmod(idx, cols)
        y = r * (h + divider)
        x = c * (w + divider)
        grid[y:y + h, x:x + w] = img
    return grid


def _find_image_in_blocks(blocks: list[OutputBlock]) -> np.ndarray | None:
    for block in blocks:
        if isinstance(block, ImageBlock):
            return block.array
        if isinstance(block, GroupBlock):
            nested = _find_image_in_blocks(block.children)
            if nested is not None:
                return nested
    return None


def _apply_image_carry(
    raw_blocks: list[OutputBlock],
    last_image: np.ndarray | None,
) -> tuple[list[OutputBlock], np.ndarray | None]:
    """Prepend a dimmed carry-forward image when the step has no image output."""

    image_to_carry = _find_image_in_blocks(raw_blocks)
    if image_to_carry is not None:
        return raw_blocks, image_to_carry
    if last_image is not None:
        return [ImageBlock(title="↑ previous", array=last_image, dim=True)] + raw_blocks, last_image
    return raw_blocks, None


def _build_span_metadata(span: StepSpan) -> dict[str, Any]:
    metadata = dict(span.operator_config)
    metadata.update({f"attributes.{key}": value for key, value in span.attributes.items()})
    return metadata
