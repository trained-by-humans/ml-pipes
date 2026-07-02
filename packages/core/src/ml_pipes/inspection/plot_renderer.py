from __future__ import annotations

import numpy as np

from .views import (
    GroupBlock,
    ImageBlock,
    OutputBlock,
    StepView,
    _flatten_step_views,
)


def _load_matplotlib() -> tuple[object, object]:
    try:
        import matplotlib
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on optional dependency state
        raise ImportError(
            "ml_pipes.inspection plotting requires matplotlib from the optional inspection extra. "
            "Install it with `pip install ml-pipes[inspection]`."
        ) from exc
    return matplotlib, plt


class PlotRenderer:
    """Renders a list of StepViews as a matplotlib Figure.

    Example::

        views = PipelineInspector().build_views(result)
        fig = PlotRenderer(cols=4).render(views)
        fig.savefig("steps.png", dpi=150)
    """

    def __init__(self, cols: int = 6, cell_w: float = 2.6, cell_h: float = 3.2) -> None:
        self.cols = cols
        self.cell_w = cell_w
        self.cell_h = cell_h

    def _block_to_lines(self, block: OutputBlock, indent: int = 0) -> list[str]:
        prefix = "  " * indent
        if isinstance(block, ImageBlock):
            return [prefix + block.title]

        if isinstance(block, GroupBlock):
            lines = [prefix + block.title]
            if not block.children:
                lines.append(prefix + "  empty")
                return lines
            for child in block.children:
                lines.extend(self._block_to_lines(child, indent + 1))
            return lines

        lines = [prefix + block.title] if block.title else []
        lines.extend(
            prefix + "  " + (f"{key}: {value}" if key else value)
            for key, value in block.rows
        )
        return lines

    def render(self, views: list[StepView]) -> "matplotlib.figure.Figure":
        matplotlib, plt = _load_matplotlib()

        flat = _flatten_step_views(views)
        count = len(flat)
        rows = max(1, (count + self.cols - 1) // self.cols)
        fig, axes = plt.subplots(
            rows,
            self.cols,
            figsize=(self.cols * self.cell_w, rows * self.cell_h),
        )
        flat_axes: list[matplotlib.axes.Axes] = np.array(axes).flatten().tolist()

        for index, (view, depth) in enumerate(flat):
            self._render_axes(flat_axes[index], view, depth)
        for ax in flat_axes[count:]:
            ax.set_visible(False)

        fig.tight_layout(pad=0.5)
        return fig

    def _render_axes(self, ax: "matplotlib.axes.Axes", view: StepView, depth: int = 0) -> None:
        ax.set_xticks([])
        ax.set_yticks([])

        if depth > 0:
            for spine in ax.spines.values():
                spine.set_edgecolor("#aaa")
                spine.set_linewidth(0.8)
                spine.set_linestyle("dashed")

        if view.error:
            for spine in ax.spines.values():
                spine.set_edgecolor("#c00")
                spine.set_linewidth(2)
                spine.set_linestyle("solid")
            ax.set_title("  " * depth + view.label, fontsize=7.5, fontweight="bold", pad=3, loc="left")
            return

        for block in view.blocks:
            if isinstance(block, ImageBlock):
                ax.imshow(block.array, alpha=0.2 if block.dim else 1.0)
                continue

            text = "\n".join(self._block_to_lines(block))
            ax.text(
                0.04,
                0.97,
                text,
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=6.5,
                family="monospace",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.85),
            )

        cfg = view.operator_config
        cfg_short = ""
        if cfg:
            joined = ", ".join(f"{key}={value!r}" for key, value in cfg.items())
            cfg_short = "\n" + (joined if len(joined) <= 30 else joined[:27] + "…")

        ax.set_title(
            "  " * depth + view.label + cfg_short,
            fontsize=7.5,
            fontweight="bold",
            pad=3,
            loc="left",
        )
