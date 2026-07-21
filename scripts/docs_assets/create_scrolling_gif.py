from __future__ import annotations

import argparse
import importlib
from pathlib import Path
import sys

_DEFAULT_HORIZONTAL_VIEWPORT_WIDTH = 1600
_DEFAULT_VERTICAL_VIEWPORT_HEIGHT = 900
_DIRECTIONS = ("horizontal", "vertical")
_DOCS_ASSET_REQUIREMENTS_FILE = Path(__file__).resolve().with_name("requirements.txt")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a scrolling GIF by panning a viewport across a larger source image. "
            "Useful for README previews of wide inspection screenshots."
        )
    )
    parser.add_argument(
        "source",
        type=Path,
        help="Source image to pan across.",
    )
    parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        default=None,
        help="Output GIF path. Defaults to <source>_scroll.gif beside the source image.",
    )
    parser.add_argument(
        "--direction",
        choices=_DIRECTIONS,
        default="horizontal",
        help="Pan direction (default: horizontal).",
    )
    parser.add_argument(
        "--viewport-width",
        type=int,
        default=None,
        metavar="PX",
        help=(
            "Crop-window width in source pixels. Controls how much of the source is visible "
            "per frame, not the final GIF width. When omitted and both output dimensions are "
            "provided, horizontal scrolling derives it from the output aspect ratio. Otherwise "
            "it defaults to 1600 for horizontal motion or the source width."
        ),
    )
    parser.add_argument(
        "--viewport-height",
        type=int,
        default=None,
        metavar="PX",
        help=(
            "Crop-window height in source pixels. Controls how much of the source is visible "
            "per frame, not the final GIF height. When omitted and both output dimensions are "
            "provided, vertical scrolling derives it from the output aspect ratio. Otherwise it "
            "defaults to the source height for horizontal motion or 900 for vertical motion."
        ),
    )
    parser.add_argument(
        "--output-width",
        type=int,
        default=None,
        metavar="PX",
        help=(
            "Final GIF width in output pixels. If only one output dimension is provided, the "
            "other is inferred to preserve the viewport aspect ratio. Provide both output "
            "dimensions when you want the output aspect ratio to define the viewport."
        ),
    )
    parser.add_argument(
        "--output-height",
        type=int,
        default=None,
        metavar="PX",
        help=(
            "Final GIF height in output pixels. If only one output dimension is provided, the "
            "other is inferred to preserve the viewport aspect ratio. Provide both output "
            "dimensions when you want the output aspect ratio to define the viewport."
        ),
    )
    parser.add_argument(
        "--step-px",
        type=int,
        default=40,
        metavar="PX",
        help="Scroll step between frames in source pixels (default: 40).",
    )
    parser.add_argument(
        "--frame-duration-ms",
        type=int,
        default=80,
        metavar="MS",
        help="Base duration of each frame in milliseconds (default: 80).",
    )
    parser.add_argument(
        "--pause-ms",
        type=int,
        default=1000,
        metavar="MS",
        help="Extra pause at the start and turnaround point in milliseconds (default: 1000).",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=None,
        metavar="N",
        help=(
            "Uniform output scale applied after cropping each frame. Prefer --output-width or "
            "--output-height when you want to target a final README-friendly size directly."
        ),
    )
    parser.add_argument(
        "--loop",
        type=int,
        default=0,
        metavar="N",
        help="GIF loop count passed to Pillow (default: 0 for infinite).",
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Enable Pillow GIF optimization.",
    )
    bounce = parser.add_mutually_exclusive_group()
    bounce.add_argument(
        "--bounce",
        dest="bounce",
        action="store_true",
        default=False,
        help="Bounce back to the starting edge after reaching the far edge.",
    )
    bounce.add_argument(
        "--no-bounce",
        dest="bounce",
        action="store_false",
        help="Scroll one-way and loop back to the start (default).",
    )
    return parser.parse_args()


def _require_positive_int(value: int, *, name: str) -> int:
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value


def _require_non_negative_int(value: int, *, name: str) -> int:
    if value < 0:
        raise ValueError(f"{name} must be greater than or equal to zero")
    return value


def _require_positive_float(value: float, *, name: str) -> float:
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value


def _require_pillow():
    try:
        return importlib.import_module("PIL.Image")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Scrolling GIF creation requires Pillow. "
            f"Install the docs-asset dependencies with: python -m pip install -r {_DOCS_ASSET_REQUIREMENTS_FILE}"
        ) from exc


def _resolve_viewport(
    image_width: int,
    image_height: int,
    *,
    direction: str,
    viewport_width: int | None,
    viewport_height: int | None,
    output_width: int | None,
    output_height: int | None,
) -> tuple[int, int]:
    width = viewport_width
    height = viewport_height
    aspect_ratio = None

    if output_width is not None:
        _require_positive_int(output_width, name="output_width")
    if output_height is not None:
        _require_positive_int(output_height, name="output_height")

    if output_width is not None and output_height is not None:
        aspect_ratio = output_width / output_height

    if direction == "horizontal":
        if height is None:
            height = image_height
        if width is None and aspect_ratio is not None:
            width = round(height * aspect_ratio)
        if width is None:
            width = min(image_width, _DEFAULT_HORIZONTAL_VIEWPORT_WIDTH)
    else:
        if width is None:
            width = image_width
        if height is None and aspect_ratio is not None:
            height = round(width / aspect_ratio)
        if height is None:
            height = min(image_height, _DEFAULT_VERTICAL_VIEWPORT_HEIGHT)

    _require_positive_int(width, name="viewport_width")
    _require_positive_int(height, name="viewport_height")

    if width > image_width:
        if aspect_ratio is not None and viewport_width is None and direction == "horizontal":
            raise ValueError(
                "Derived viewport_width exceeds the source width. "
                "Use a taller output aspect ratio, provide a smaller viewport_height, "
                "or set viewport_width explicitly."
            )
        raise ValueError(f"viewport_width ({width}) cannot exceed source width ({image_width})")
    if height > image_height:
        if aspect_ratio is not None and viewport_height is None and direction == "vertical":
            raise ValueError(
                "Derived viewport_height exceeds the source height. "
                "Use a wider output aspect ratio, provide a smaller viewport_width, "
                "or set viewport_height explicitly."
            )
        raise ValueError(f"viewport_height ({height}) cannot exceed source height ({image_height})")

    return width, height


def _resolve_output_size(
    viewport_width: int,
    viewport_height: int,
    *,
    output_width: int | None,
    output_height: int | None,
    scale: float | None,
) -> tuple[int, int]:
    if output_width is not None:
        _require_positive_int(output_width, name="output_width")
    if output_height is not None:
        _require_positive_int(output_height, name="output_height")

    if scale is not None:
        _require_positive_float(scale, name="scale")

    if scale is not None and (output_width is not None or output_height is not None):
        raise ValueError("scale cannot be combined with output_width or output_height")

    if output_width is None and output_height is None:
        if scale is None:
            return viewport_width, viewport_height
        return (
            max(1, round(viewport_width * scale)),
            max(1, round(viewport_height * scale)),
        )

    if output_width is not None and output_height is not None:
        expected_height = max(1, round(output_width * viewport_height / viewport_width))
        expected_width = max(1, round(output_height * viewport_width / viewport_height))
        if output_height != expected_height and output_width != expected_width:
            raise ValueError(
                "output_width/output_height must preserve the viewport aspect ratio. "
                "Specify only one output dimension or adjust the viewport."
            )
        return output_width, output_height

    if output_width is not None:
        return output_width, max(1, round(output_width * viewport_height / viewport_width))

    assert output_height is not None
    return max(1, round(output_height * viewport_width / viewport_height)), output_height


def _resolve_output_path(source_path: Path, output: Path | str | None) -> Path:
    if output is None:
        return source_path.with_name(f"{source_path.stem}_scroll.gif")
    return Path(output).expanduser()


def _build_animation_plan(
    max_offset: int,
    *,
    step_px: int,
    frame_duration_ms: int,
    pause_ms: int,
    bounce: bool,
) -> tuple[list[int], list[int]]:
    _require_non_negative_int(max_offset, name="max_offset")
    _require_positive_int(step_px, name="step_px")
    _require_positive_int(frame_duration_ms, name="frame_duration_ms")
    _require_non_negative_int(pause_ms, name="pause_ms")

    if max_offset == 0:
        return [0], [frame_duration_ms + pause_ms]

    offsets = list(range(0, max_offset + 1, step_px))
    if offsets[-1] != max_offset:
        offsets.append(max_offset)

    if bounce:
        offsets = offsets + offsets[-2::-1]

    durations = [frame_duration_ms] * len(offsets)
    durations[0] += pause_ms
    turnaround_index = offsets.index(max_offset)
    durations[turnaround_index] += pause_ms
    return offsets, durations


def _crop_box(
    *,
    direction: str,
    offset: int,
    viewport_width: int,
    viewport_height: int,
) -> tuple[int, int, int, int]:
    if direction == "horizontal":
        return (offset, 0, offset + viewport_width, viewport_height)
    return (0, offset, viewport_width, offset + viewport_height)


def create_scrolling_gif(
    source: Path | str,
    output: Path | str | None = None,
    *,
    direction: str = "horizontal",
    viewport_width: int | None = None,
    viewport_height: int | None = None,
    output_width: int | None = None,
    output_height: int | None = None,
    step_px: int = 40,
    frame_duration_ms: int = 80,
    pause_ms: int = 1000,
    scale: float | None = None,
    loop: int = 0,
    bounce: bool = False,
    optimize: bool = False,
) -> Path:
    if direction not in _DIRECTIONS:
        raise ValueError(f"direction must be one of {_DIRECTIONS}")

    _require_non_negative_int(loop, name="loop")

    image_module = _require_pillow()
    resampling = getattr(image_module, "Resampling", image_module)
    source_path = Path(source).expanduser()
    if not source_path.is_file():
        raise ValueError(f"Source image does not exist: {source_path}")

    output_path = _resolve_output_path(source_path, output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with image_module.open(source_path) as image:
        base = image.convert("RGBA")
        image_width, image_height = base.size
        viewport_width, viewport_height = _resolve_viewport(
            image_width,
            image_height,
            direction=direction,
            viewport_width=viewport_width,
            viewport_height=viewport_height,
            output_width=output_width,
            output_height=output_height,
        )
        output_width, output_height = _resolve_output_size(
            viewport_width,
            viewport_height,
            output_width=output_width,
            output_height=output_height,
            scale=scale,
        )

        motion_extent = image_width - viewport_width if direction == "horizontal" else image_height - viewport_height
        if motion_extent <= 0:
            raise ValueError(
                "Viewport does not leave any room to scroll. "
                "Use a smaller viewport on the scrolling axis."
            )

        offsets, durations = _build_animation_plan(
            motion_extent,
            step_px=step_px,
            frame_duration_ms=frame_duration_ms,
            pause_ms=pause_ms,
            bounce=bounce,
        )

        frames = []
        for offset in offsets:
            frame = base.crop(
                _crop_box(
                    direction=direction,
                    offset=offset,
                    viewport_width=viewport_width,
                    viewport_height=viewport_height,
                )
            )
            if frame.size != (output_width, output_height):
                frame = frame.resize((output_width, output_height), resampling.LANCZOS)
            frames.append(frame.convert("P", palette=resampling.ADAPTIVE if hasattr(resampling, "ADAPTIVE") else image_module.ADAPTIVE))

    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=loop,
        optimize=optimize,
        disposal=2,
    )
    return output_path


def main() -> int:
    args = _parse_args()
    output = create_scrolling_gif(
        args.source,
        args.output,
        direction=args.direction,
        viewport_width=args.viewport_width,
        viewport_height=args.viewport_height,
        output_width=args.output_width,
        output_height=args.output_height,
        step_px=args.step_px,
        frame_duration_ms=args.frame_duration_ms,
        pause_ms=args.pause_ms,
        scale=args.scale,
        loop=args.loop,
        bounce=args.bounce,
        optimize=args.optimize,
    )
    print(f"Saved scrolling GIF: {output}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1) from None
