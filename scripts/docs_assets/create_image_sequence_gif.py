from __future__ import annotations

import argparse
import importlib
from pathlib import Path
import sys


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create an animated GIF from 2 or more source images. "
            "Useful for before/after states like tooltip or tile interactions."
        )
    )
    parser.add_argument(
        "sources",
        type=Path,
        nargs="+",
        help="Source images in display order.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output GIF path. Defaults to <first_source>_sequence.gif beside the first source image.",
    )
    parser.add_argument(
        "--output-width",
        type=int,
        default=None,
        metavar="PX",
        help=(
            "Final GIF width in output pixels. If only one output dimension is provided, the "
            "other is inferred to preserve the source aspect ratio."
        ),
    )
    parser.add_argument(
        "--output-height",
        type=int,
        default=None,
        metavar="PX",
        help=(
            "Final GIF height in output pixels. If only one output dimension is provided, the "
            "other is inferred to preserve the source aspect ratio."
        ),
    )
    parser.add_argument(
        "--frame-duration-ms",
        type=int,
        default=700,
        metavar="MS",
        help="Base duration of each frame in milliseconds (default: 700).",
    )
    parser.add_argument(
        "--pause-ms",
        type=int,
        default=1000,
        metavar="MS",
        help="Extra pause at the first frame and the last source frame (default: 1000).",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=None,
        metavar="N",
        help=(
            "Uniform output scale applied to every source image. Prefer --output-width or "
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
        help="Play forward, then reverse back to the first image before looping.",
    )
    bounce.add_argument(
        "--no-bounce",
        dest="bounce",
        action="store_false",
        help="Play forward once, then jump back to the first image on loop (default).",
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
            "Image-sequence GIF creation requires Pillow. "
            "Install it with: python -m pip install pillow"
        ) from exc


def _resolve_output_size(
    source_width: int,
    source_height: int,
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
            return source_width, source_height
        return (
            max(1, round(source_width * scale)),
            max(1, round(source_height * scale)),
        )

    if output_width is not None and output_height is not None:
        expected_height = max(1, round(output_width * source_height / source_width))
        expected_width = max(1, round(output_height * source_width / source_height))
        if output_height != expected_height and output_width != expected_width:
            raise ValueError(
                "output_width/output_height must preserve the source aspect ratio. "
                "Specify only one output dimension when you want it inferred."
            )
        return output_width, output_height

    if output_width is not None:
        return output_width, max(1, round(output_width * source_height / source_width))

    assert output_height is not None
    return max(1, round(output_height * source_width / source_height)), output_height


def _resolve_output_path(first_source_path: Path, output: Path | str | None) -> Path:
    if output is None:
        return first_source_path.with_name(f"{first_source_path.stem}_sequence.gif")
    return Path(output).expanduser()


def create_image_sequence_gif(
    sources: list[Path | str],
    output: Path | str | None = None,
    *,
    output_width: int | None = None,
    output_height: int | None = None,
    frame_duration_ms: int = 700,
    pause_ms: int = 1000,
    scale: float | None = None,
    loop: int = 0,
    bounce: bool = False,
    optimize: bool = False,
) -> Path:
    if len(sources) < 2:
        raise ValueError("Provide at least 2 source images")

    _require_positive_int(frame_duration_ms, name="frame_duration_ms")
    _require_non_negative_int(pause_ms, name="pause_ms")
    _require_non_negative_int(loop, name="loop")

    image_module = _require_pillow()
    resampling = getattr(image_module, "Resampling", image_module)
    source_paths = [Path(source).expanduser() for source in sources]
    for source_path in source_paths:
        if not source_path.is_file():
            raise ValueError(f"Source image does not exist: {source_path}")

    output_path = _resolve_output_path(source_paths[0], output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    base_size: tuple[int, int] | None = None
    source_frames = []
    for source_path in source_paths:
        with image_module.open(source_path) as image:
            frame = image.convert("RGBA")
            if base_size is None:
                base_size = frame.size
            elif frame.size != base_size:
                raise ValueError(
                    "All source images must have the same size. "
                    "Capture them with the same viewport or resize them first."
                )
            source_frames.append(frame)

    assert base_size is not None
    output_size = _resolve_output_size(
        base_size[0],
        base_size[1],
        output_width=output_width,
        output_height=output_height,
        scale=scale,
    )

    animation_frames = source_frames + source_frames[-2::-1] if bounce else source_frames
    durations = [frame_duration_ms] * len(animation_frames)
    durations[0] += pause_ms
    durations[len(source_frames) - 1] += pause_ms

    rendered_frames = []
    for frame in animation_frames:
        rendered = frame
        if rendered.size != output_size:
            rendered = rendered.resize(output_size, resampling.LANCZOS)
        rendered_frames.append(
            rendered.convert(
                "P",
                palette=resampling.ADAPTIVE if hasattr(resampling, "ADAPTIVE") else image_module.ADAPTIVE,
            )
        )

    rendered_frames[0].save(
        output_path,
        save_all=True,
        append_images=rendered_frames[1:],
        duration=durations,
        loop=loop,
        optimize=optimize,
        disposal=2,
    )
    return output_path


def main() -> int:
    args = _parse_args()
    output = create_image_sequence_gif(
        args.sources,
        output=args.output,
        output_width=args.output_width,
        output_height=args.output_height,
        frame_duration_ms=args.frame_duration_ms,
        pause_ms=args.pause_ms,
        scale=args.scale,
        loop=args.loop,
        bounce=args.bounce,
        optimize=args.optimize,
    )
    print(f"Saved image-sequence GIF: {output}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1) from None
