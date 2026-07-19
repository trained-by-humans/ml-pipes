from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys

from PIL import Image
import pytest


def _load_create_scrolling_gif_module():
    module_path = Path(__file__).resolve().parents[3] / "scripts" / "docs_assets" / "create_scrolling_gif.py"
    spec = importlib.util.spec_from_file_location("create_scrolling_gif", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _make_source_image(path: Path, *, width: int = 240, height: int = 60) -> None:
    image = Image.new("RGB", (width, height))
    for x in range(width):
        color = (x % 256, (x * 3) % 256, (255 - x) % 256)
        for y in range(height):
            image.putpixel((x, y), color)
    image.save(path)


def test_build_animation_plan_bounces_and_pauses_at_edges() -> None:
    module = _load_create_scrolling_gif_module()

    offsets, durations = module._build_animation_plan(
        100,
        step_px=40,
        frame_duration_ms=80,
        pause_ms=1000,
        bounce=True,
    )

    assert offsets == [0, 40, 80, 100, 80, 40, 0]
    assert durations == [1080, 80, 80, 1080, 80, 80, 80]


def test_parse_args_defaults_to_no_bounce(monkeypatch) -> None:
    module = _load_create_scrolling_gif_module()
    monkeypatch.setattr(sys, "argv", ["create_scrolling_gif.py", "wide.png"])

    args = module._parse_args()

    assert args.bounce is False


def test_create_scrolling_gif_writes_multiple_frames(tmp_path: Path) -> None:
    module = _load_create_scrolling_gif_module()
    source = tmp_path / "wide.png"
    output = tmp_path / "scroll.gif"
    _make_source_image(source)

    saved = module.create_scrolling_gif(
        source,
        output,
        viewport_width=80,
        viewport_height=60,
        output_width=40,
        step_px=40,
        frame_duration_ms=50,
        pause_ms=200,
        bounce=True,
    )

    assert saved == output
    assert output.is_file()

    with Image.open(output) as gif:
        assert gif.format == "GIF"
        assert gif.n_frames == 9
        assert gif.size == (40, 30)
        gif.seek(0)
        assert gif.info["duration"] == 250
        gif.seek(4)
        assert gif.info["duration"] == 250


def test_create_scrolling_gif_derives_horizontal_viewport_from_output_ratio(tmp_path: Path) -> None:
    module = _load_create_scrolling_gif_module()
    source = tmp_path / "wide.png"
    output = tmp_path / "scroll.gif"
    _make_source_image(source)

    saved = module.create_scrolling_gif(
        source,
        output,
        output_width=40,
        output_height=20,
        step_px=60,
        frame_duration_ms=50,
        pause_ms=200,
    )

    assert saved == output
    assert output.is_file()

    with Image.open(output) as gif:
        assert gif.size == (40, 20)
        assert gif.n_frames == 3
        gif.seek(0)
        assert gif.info["duration"] == 250
        gif.seek(2)
        assert gif.info["duration"] == 250


def test_create_scrolling_gif_defaults_output_next_to_source(tmp_path: Path) -> None:
    module = _load_create_scrolling_gif_module()
    source = tmp_path / "wide.png"
    _make_source_image(source)

    saved = module.create_scrolling_gif(
        source,
        viewport_width=80,
        viewport_height=60,
        output_width=40,
        step_px=40,
        frame_duration_ms=50,
        pause_ms=200,
    )

    assert saved == tmp_path / "wide_scroll.gif"
    assert saved.is_file()


def test_create_scrolling_gif_rejects_mismatched_output_aspect_ratio(tmp_path: Path) -> None:
    module = _load_create_scrolling_gif_module()
    source = tmp_path / "wide.png"
    output = tmp_path / "scroll.gif"
    _make_source_image(source)

    with pytest.raises(ValueError, match="preserve the viewport aspect ratio"):
        module.create_scrolling_gif(
            source,
            output,
            viewport_width=80,
            viewport_height=60,
            output_width=50,
            output_height=50,
        )


def test_main_prints_saved_path(monkeypatch, tmp_path: Path, capsys) -> None:
    module = _load_create_scrolling_gif_module()
    output = tmp_path / "wide_scroll.gif"

    monkeypatch.setattr(
        module,
        "_parse_args",
        lambda: argparse.Namespace(
            source=Path("wide.png"),
            output=None,
            direction="horizontal",
            viewport_width=80,
            viewport_height=60,
            output_width=40,
            output_height=None,
            step_px=40,
            frame_duration_ms=80,
            pause_ms=1000,
            scale=None,
            loop=0,
            bounce=True,
            optimize=False,
        ),
    )
    monkeypatch.setattr(module, "create_scrolling_gif", lambda *_args, **_kwargs: output)

    assert module.main() == 0
    assert capsys.readouterr().out.strip() == f"Saved scrolling GIF: {output}"
