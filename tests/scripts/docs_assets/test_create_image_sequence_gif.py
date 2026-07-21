from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys

from PIL import Image
import pytest


def _load_create_image_sequence_gif_module():
    module_path = Path(__file__).resolve().parents[3] / "scripts" / "docs_assets" / "create_image_sequence_gif.py"
    spec = importlib.util.spec_from_file_location("create_image_sequence_gif", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _make_source_image(path: Path, *, color: tuple[int, int, int], width: int = 120, height: int = 60) -> None:
    Image.new("RGB", (width, height), color).save(path)


def test_parse_args_defaults_to_no_bounce(monkeypatch) -> None:
    module = _load_create_image_sequence_gif_module()
    monkeypatch.setattr(sys, "argv", ["create_image_sequence_gif.py", "before.png", "after.png"])

    args = module._parse_args()

    assert args.output is None
    assert args.bounce is False
    assert args.sources == [Path("before.png"), Path("after.png")]


def test_create_image_sequence_gif_writes_multiple_frames(tmp_path: Path) -> None:
    module = _load_create_image_sequence_gif_module()
    before = tmp_path / "before.png"
    after = tmp_path / "after.png"
    output = tmp_path / "tooltip.gif"
    _make_source_image(before, color=(255, 0, 0))
    _make_source_image(after, color=(0, 255, 0))

    saved = module.create_image_sequence_gif(
        [before, after],
        output=output,
        output_width=60,
        frame_duration_ms=100,
        pause_ms=200,
    )

    assert saved == output
    assert output.is_file()

    with Image.open(output) as gif:
        assert gif.format == "GIF"
        assert gif.n_frames == 2
        assert gif.size == (60, 30)
        gif.seek(0)
        assert gif.info["duration"] == 300
        gif.seek(1)
        assert gif.info["duration"] == 300


def test_create_image_sequence_gif_bounces_back_when_enabled(tmp_path: Path) -> None:
    module = _load_create_image_sequence_gif_module()
    first = tmp_path / "01.png"
    second = tmp_path / "02.png"
    third = tmp_path / "03.png"
    output = tmp_path / "sequence.gif"
    _make_source_image(first, color=(255, 0, 0))
    _make_source_image(second, color=(0, 255, 0))
    _make_source_image(third, color=(0, 0, 255))

    saved = module.create_image_sequence_gif(
        [first, second, third],
        output=output,
        frame_duration_ms=80,
        pause_ms=200,
        bounce=True,
    )

    assert saved == output

    with Image.open(output) as gif:
        assert gif.n_frames == 5
        gif.seek(0)
        assert gif.info["duration"] == 280
        gif.seek(2)
        assert gif.info["duration"] == 280
        gif.seek(4)
        assert gif.info["duration"] == 80


def test_create_image_sequence_gif_defaults_output_next_to_first_source(tmp_path: Path) -> None:
    module = _load_create_image_sequence_gif_module()
    before = tmp_path / "before.png"
    after = tmp_path / "after.png"
    _make_source_image(before, color=(255, 0, 0))
    _make_source_image(after, color=(0, 255, 0))

    saved = module.create_image_sequence_gif([before, after])

    assert saved == tmp_path / "before_sequence.gif"
    assert saved.is_file()


def test_create_image_sequence_gif_rejects_mismatched_sizes(tmp_path: Path) -> None:
    module = _load_create_image_sequence_gif_module()
    before = tmp_path / "before.png"
    after = tmp_path / "after.png"
    _make_source_image(before, color=(255, 0, 0), width=120, height=60)
    _make_source_image(after, color=(0, 255, 0), width=160, height=60)

    with pytest.raises(ValueError, match="same size"):
        module.create_image_sequence_gif([before, after])


def test_main_prints_saved_path(monkeypatch, tmp_path: Path, capsys) -> None:
    module = _load_create_image_sequence_gif_module()
    output = tmp_path / "before_sequence.gif"

    monkeypatch.setattr(
        module,
        "_parse_args",
        lambda: argparse.Namespace(
            sources=[Path("before.png"), Path("after.png")],
            output=None,
            output_width=60,
            output_height=None,
            frame_duration_ms=700,
            pause_ms=1000,
            scale=None,
            loop=0,
            bounce=False,
            optimize=False,
        ),
    )
    monkeypatch.setattr(module, "create_image_sequence_gif", lambda *_args, **_kwargs: output)

    assert module.main() == 0
    assert capsys.readouterr().out.strip() == f"Saved image-sequence GIF: {output}"
