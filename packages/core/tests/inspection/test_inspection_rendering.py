from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from ml_pipes.inspection import (
    GroupBlock,
    HtmlRenderer,
    ImageBlock,
    InspectionResult,
    PipelineInspector,
    StepView,
    TextBlock,
    ndarray_image_formatter,
    register_value_formatter,
)
from ml_pipes.inspection.registry import FormatterRegistry
from ml_pipes.tracing import StepSpan


def _inspection_result() -> InspectionResult:
    return InspectionResult(
        [
            StepSpan(
                label="0:Example",
                start_time=0.0,
                duration_s=0.01,
                output_value={"label": "spam", "msg": "hello"},
            )
        ]
    )


def _fake_mkstemp_factory(tmp_path: Path):
    counter = {"value": 0}

    def fake_mkstemp(suffix: str = "", prefix: str = "tmp", dir: str | None = None, text: bool = False) -> tuple[int, str]:
        path = tmp_path / f"{prefix}{counter['value']}{suffix}"
        counter["value"] += 1
        fd = os.open(path, os.O_CREAT | os.O_TRUNC | os.O_RDWR)
        return fd, str(path)

    return fake_mkstemp


def _saved_reports(tmp_path: Path) -> list[Path]:
    return sorted(tmp_path.glob("ml_pipes_inspect_*.html"))


def test_pipeline_inspector_render_defaults_to_horizontal_orientation():
    html = PipelineInspector().render(_inspection_result())

    assert '<div class="insp-container insp-container--horizontal">' in html


def test_pipeline_inspector_can_register_bgr_ndarray_image_formatter() -> None:
    inspector = PipelineInspector().register_value_formatter(
        np.ndarray,
        ndarray_image_formatter(default_color_space="BGR"),
    )
    result = InspectionResult(
        [
            StepSpan(
                label="0:identity",
                start_time=0.0,
                duration_s=0.01,
                output_value=np.full((2, 3, 3), (0, 0, 255), dtype=np.uint8),
            )
        ]
    )

    block = inspector.build_views(result)[0].blocks[0]

    assert isinstance(block, ImageBlock)
    assert block.title == "ndarray  3×2  BGR"
    assert np.array_equal(block.array[0, 0], [255, 0, 0])


def test_global_value_formatter_applies_to_pipeline_inspectors() -> None:
    class Packet:
        pass

    register_value_formatter(Packet, lambda _value: [TextBlock("packet", [("source", "global")])])

    blocks = PipelineInspector()._value_to_blocks(Packet())

    assert blocks == [TextBlock("packet", [("source", "global")])]


def test_pipeline_inspector_render_supports_vertical_orientation():
    html = PipelineInspector().render(_inspection_result(), orientation="vertical")

    assert '<div class="insp-container insp-container--vertical">' in html


def test_pipeline_inspector_render_normalizes_orientation():
    html = PipelineInspector().render(_inspection_result(), orientation=" Vertical ")

    assert '<div class="insp-container insp-container--vertical">' in html


def test_pipeline_inspector_show_delegates_to_renderer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inspector = PipelineInspector()
    shown: dict[str, object] = {}

    def fake_show(views: list[StepView], orientation: str = "horizontal") -> None:
        shown["labels"] = [view.label for view in views]
        shown["orientation"] = orientation

    monkeypatch.setattr(inspector._renderer, "show", fake_show)

    result = _inspection_result()
    inspector.show(result, orientation=" Vertical ")

    assert shown == {
        "labels": ["0:Example"],
        "orientation": "vertical",
    }


def test_html_renderer_show_displays_html_in_jupyter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shown: dict[str, object] = {}

    fake_display_module = ModuleType("IPython.display")

    def fake_html(value: str) -> str:
        shown["html"] = value
        return value

    def fake_display(value: object) -> None:
        shown["displayed"] = value

    fake_display_module.HTML = fake_html
    fake_display_module.display = fake_display
    fake_ipython_module = ModuleType("IPython")
    fake_ipython_module.display = fake_display_module

    monkeypatch.setitem(sys.modules, "IPython", fake_ipython_module)
    monkeypatch.setitem(sys.modules, "IPython.display", fake_display_module)
    monkeypatch.setattr("ml_pipes.inspection.html_renderer._IN_JUPYTER", True)
    monkeypatch.setattr(
        HtmlRenderer,
        "render",
        lambda self, views, orientation="horizontal": f"<html data-orientation='{orientation}'></html>",
    )

    HtmlRenderer().show([], orientation="vertical")

    assert shown == {
        "html": "<html data-orientation='vertical'></html>",
        "displayed": "<html data-orientation='vertical'></html>",
    }


def test_html_renderer_show_announces_saved_report_and_browser_open_outside_jupyter(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr("ml_pipes.inspection.html_renderer._IN_JUPYTER", False)
    monkeypatch.setattr(
        "ml_pipes.inspection.html_renderer.tempfile.mkstemp",
        _fake_mkstemp_factory(tmp_path),
    )

    opened: dict[str, str] = {}

    def fake_open(uri: str) -> bool:
        opened["uri"] = uri
        return True

    monkeypatch.setattr("ml_pipes.inspection.html_renderer.webbrowser.open", fake_open)

    HtmlRenderer().show(
        [StepView(label="0:Example", operator_config={}, blocks=[TextBlock("dict", [("label", "spam")])])]
    )
    captured = capsys.readouterr()

    reports = _saved_reports(tmp_path)
    assert len(reports) == 1
    report = reports[0]
    assert report.exists()
    assert opened["uri"] == report.as_uri()
    assert f"Inspection report saved to: {report}" in captured.err
    assert "Opening inspection report in browser..." in captured.err
    assert "Browser launch was not confirmed." not in captured.err
    assert "<title>Pipeline inspection</title>" in report.read_text(encoding="utf-8")


def test_html_renderer_show_warns_when_browser_launch_is_not_confirmed_outside_jupyter(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr("ml_pipes.inspection.html_renderer._IN_JUPYTER", False)
    monkeypatch.setattr(
        "ml_pipes.inspection.html_renderer.tempfile.mkstemp",
        _fake_mkstemp_factory(tmp_path),
    )
    monkeypatch.setattr("ml_pipes.inspection.html_renderer.webbrowser.open", lambda _uri: False)

    HtmlRenderer().show(
        [StepView(label="0:Example", operator_config={}, blocks=[TextBlock("dict", [("label", "spam")])])]
    )
    captured = capsys.readouterr()
    reports = _saved_reports(tmp_path)

    assert len(reports) == 1
    report = reports[0]
    assert report.exists()
    assert f"Inspection report saved to: {report}" in captured.err
    assert "Opening inspection report in browser..." in captured.err
    assert "Browser launch was not confirmed. If nothing opened, use the saved report path above." in captured.err


def test_html_renderer_rejects_unknown_orientation():
    with pytest.raises(ValueError, match="Invalid inspection orientation"):
        HtmlRenderer().render([], orientation="diagonal")


def test_html_renderer_vertical_layout_widens_cards_for_tabular_output():
    html = HtmlRenderer().render(
        [StepView(label="0:Example", operator_config={}, blocks=[TextBlock("dict", [("a", "b")])])],
        orientation="vertical",
    )

    assert '<div class="insp-container insp-container--vertical">' in html
    assert "width: calc(100% - 8px);" in html


@dataclass
class _Stats:
    scanned: int
    kept: int


@dataclass
class _Run:
    stats: _Stats
    started_at: float


def test_pipeline_inspector_formats_nested_dataclass_as_group_blocks():
    blocks = PipelineInspector()._value_to_blocks(_Run(stats=_Stats(scanned=0, kept=1), started_at=2.5))

    assert len(blocks) == 1
    assert isinstance(blocks[0], GroupBlock)
    assert blocks[0].title == "_Run"
    assert [child.title for child in blocks[0].children] == ["stats: _Stats", ""]

    stats_group = blocks[0].children[0]
    assert isinstance(stats_group, GroupBlock)
    assert [child.title for child in stats_group.children] == ["", ""]
    assert all(isinstance(child, TextBlock) for child in stats_group.children)
    assert stats_group.children[0].rows == [("scanned", "0")]
    assert stats_group.children[1].rows == [("kept", "1")]

    started_at = blocks[0].children[1]
    assert isinstance(started_at, TextBlock)
    assert started_at.rows == [("started_at", "2.5")]


def test_pipeline_inspector_formats_mapping_as_group_blocks():
    blocks = PipelineInspector()._value_to_blocks({"stats": {"kept": 1}, "started_at": 2.5})

    assert len(blocks) == 1
    assert isinstance(blocks[0], GroupBlock)
    assert blocks[0].title == "dict"
    assert [child.title for child in blocks[0].children] == ["stats: dict", ""]

    stats_group = blocks[0].children[0]
    assert isinstance(stats_group, GroupBlock)
    assert [child.title for child in stats_group.children] == [""]
    assert isinstance(stats_group.children[0], TextBlock)
    assert stats_group.children[0].rows == [("kept", "1")]

    started_at = blocks[0].children[1]
    assert isinstance(started_at, TextBlock)
    assert started_at.rows == [("started_at", "2.5")]


def test_pipeline_inspector_keeps_full_scalar_text_before_compaction():
    text = "x" * 121

    blocks = PipelineInspector()._value_to_blocks(text)

    assert blocks == [TextBlock("str", [("", text)])]


def test_pipeline_inspector_build_views_trims_long_scalar_text_during_compaction():
    text = "x" * 121
    result = InspectionResult(
        [
            StepSpan(
                label="0:Example",
                start_time=0.0,
                duration_s=0.01,
                output_value=text,
            )
        ]
    )

    views = PipelineInspector().build_views(result)

    assert views[0].blocks == [TextBlock("str", [("", ("x" * 120) + "…")])]


def test_pipeline_inspector_keeps_all_mapping_members_before_compaction():
    payload = {f"k{i}": i for i in range(13)}

    blocks = PipelineInspector()._value_to_blocks(payload)

    assert len(blocks) == 1
    assert isinstance(blocks[0], GroupBlock)
    assert blocks[0].title == "dict"
    assert len(blocks[0].children) == 13
    assert all(child.title != "…" for child in blocks[0].children)


def test_pipeline_inspector_build_views_trims_large_mapping_during_compaction():
    payload = {f"k{i}": i for i in range(13)}
    result = InspectionResult(
        [
            StepSpan(
                label="0:Example",
                start_time=0.0,
                duration_s=0.01,
                output_value=payload,
            )
        ]
    )

    views = PipelineInspector().build_views(result)

    assert len(views) == 1
    assert len(views[0].blocks) == 1
    block = views[0].blocks[0]
    assert isinstance(block, GroupBlock)
    assert block.title == "dict"
    assert len(block.children) == 13
    assert isinstance(block.children[-1], TextBlock)
    assert block.children[-1].title == "…"
    assert block.children[-1].rows == [("", "+1 more")]


def test_pipeline_inspector_build_views_handles_cyclic_mappings():
    payload: dict[str, object] = {"label": "spam"}
    payload["self"] = payload
    result = InspectionResult(
        [
            StepSpan(
                label="0:Example",
                start_time=0.0,
                duration_s=0.01,
                output_value=payload,
            )
        ]
    )

    views = PipelineInspector().build_views(result)

    assert len(views) == 1
    assert len(views[0].blocks) == 1
    block = views[0].blocks[0]
    assert isinstance(block, GroupBlock)
    assert block.title == "dict"
    assert len(block.children) == 2
    assert isinstance(block.children[0], TextBlock)
    assert block.children[0].rows == [("label", "spam")]
    assert isinstance(block.children[1], TextBlock)
    assert block.children[1].rows == [("self", "<recursive dict>")]


def test_pipeline_inspector_formats_cyclic_lists_without_recursing_forever():
    payload: list[object] = []
    payload.append(payload)

    blocks = PipelineInspector()._value_to_blocks(payload)

    assert len(blocks) == 1
    assert isinstance(blocks[0], GroupBlock)
    assert blocks[0].title == "list[list]  ×1"
    assert len(blocks[0].children) == 1

    item = blocks[0].children[0]
    assert isinstance(item, GroupBlock)
    assert item.title == "[0]"
    assert len(item.children) == 1

    nested = item.children[0]
    assert isinstance(nested, GroupBlock)
    assert nested.title == "list[list]  ×1"
    assert len(nested.children) == 1
    assert isinstance(nested.children[0], TextBlock)
    assert nested.children[0].rows == [("", "<recursive list>")]


def test_html_renderer_renders_group_block_boundaries():
    html = HtmlRenderer().render(
        [
            StepView(
                label="0:Example",
                operator_config={},
                blocks=[
                    GroupBlock(
                        title="run: _Run",
                        children=[
                            GroupBlock(
                                title="stats: _Stats",
                                children=[TextBlock("kept", [("", "1")])],
                            )
                        ],
                    )
                ],
            )
        ],
        orientation="vertical",
    )

    assert 'class="insp-group"' in html
    assert "run: _Run" in html
    assert "stats: _Stats" in html
    assert "kept" in html


def test_html_renderer_coalesces_inline_group_rows_into_one_table():
    html = HtmlRenderer().render(
        [
            StepView(
                label="0:Example",
                operator_config={},
                blocks=[
                    GroupBlock(
                        title="run: _Run",
                        children=[
                            TextBlock("", [("items", "<generator object ...>")]),
                            TextBlock("", [("kept", "0")]),
                            GroupBlock(
                                title="stats: _Stats",
                                children=[TextBlock("", [("matched_filter", "0")])],
                            ),
                        ],
                    )
                ],
            )
        ],
        orientation="vertical",
    )

    assert html.count('class="insp-inline-grid"') == 2
    assert "items" in html
    assert "kept" in html
    assert "matched_filter" in html


def test_pipeline_inspector_maps_list_of_mappings_to_group_blocks():
    blocks = PipelineInspector()._value_to_blocks([{"label": "spam"}, {"label": "ham"}])

    assert len(blocks) == 1
    assert isinstance(blocks[0], GroupBlock)
    assert blocks[0].title == "list[dict]  ×2"
    assert [child.title for child in blocks[0].children] == ["[0]", "[1]"]

    first = blocks[0].children[0]
    assert isinstance(first, GroupBlock)
    assert len(first.children) == 1
    assert isinstance(first.children[0], GroupBlock)
    assert first.children[0].title == "dict"


def test_pipeline_inspector_build_views_summarizes_list_of_mappings_without_generic_ellipsis():
    result = InspectionResult(
        [
            StepSpan(
                label="0:Example",
                start_time=0.0,
                duration_s=0.01,
                output_value=[{"label": "spam"}, {"label": "ham"}],
            )
        ]
    )

    views = PipelineInspector().build_views(result)

    assert views[0].blocks == [TextBlock("list  ×2", [("[0]", "dict  label spam"), ("[1]", "dict  label ham")])]


def test_pipeline_inspector_build_views_summarizes_list_of_strings_with_registered_formatter():
    inspector = PipelineInspector().register_value_formatter(
        str,
        lambda value: [TextBlock("str", [("", value)])],
    )
    result = InspectionResult(
        [
            StepSpan(
                label="0:Example",
                start_time=0.0,
                duration_s=0.01,
                output_value=["hello", "world"],
            )
        ]
    )

    views = inspector.build_views(result)

    assert views[0].blocks == [TextBlock("list  ×2", [("[0]", "hello"), ("[1]", "world")])]


def test_pipeline_inspector_build_views_preserves_metadata_for_list_of_image_records():
    payload = [
        {
            "label": "cat",
            "score": 0.91,
            "crop": np.zeros((4, 4, 3), dtype=np.uint8),
        },
        {
            "label": "dog",
            "score": 0.87,
            "crop": np.zeros((4, 4, 3), dtype=np.uint8),
        },
    ]
    result = InspectionResult(
        [
            StepSpan(
                label="0:Example",
                start_time=0.0,
                duration_s=0.01,
                output_value=payload,
            )
        ]
    )

    views = PipelineInspector().build_views(result)

    assert len(views) == 1
    assert views[0].blocks == [
        TextBlock(
            "list  ×2",
            [
                ("[0]", "dict  label cat  |  score 0.91  |  +1 more"),
                ("[1]", "dict  label dog  |  score 0.87  |  +1 more"),
            ],
        )
    ]


def test_pipeline_inspector_build_views_marks_truncated_image_grid_preview():
    result = InspectionResult(
        [
            StepSpan(
                label="0:Example",
                start_time=0.0,
                duration_s=0.01,
                output_value=[np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(13)],
            )
        ]
    )

    views = PipelineInspector().build_views(result)

    assert len(views) == 1
    assert len(views[0].blocks) == 1
    block = views[0].blocks[0]
    assert isinstance(block, ImageBlock)
    assert block.title == "ndarray  ×13  (showing 12 out of 13)"


def test_pipeline_inspector_keeps_all_list_items_before_compaction():
    blocks = PipelineInspector()._value_to_blocks([{"label": str(i)} for i in range(13)])

    assert len(blocks) == 1
    assert isinstance(blocks[0], GroupBlock)
    assert blocks[0].title == "list[dict]  ×13"
    assert len(blocks[0].children) == 13
    assert [child.title for child in blocks[0].children[:3]] == ["[0]", "[1]", "[2]"]
    assert blocks[0].children[-1].title == "[12]"


def test_pipeline_inspector_build_views_trims_large_list_during_compaction():
    result = InspectionResult(
        [
            StepSpan(
                label="0:Example",
                start_time=0.0,
                duration_s=0.01,
                output_value=[{"label": str(i)} for i in range(13)],
            )
        ]
    )

    views = PipelineInspector().build_views(result)

    assert len(views) == 1
    assert len(views[0].blocks) == 1
    block = views[0].blocks[0]
    assert isinstance(block, TextBlock)
    assert block.title == "list  ×13"
    assert block.rows[-1] == ("…", "+7 more")


def test_view_blocks_and_steps_expose_summary_text():
    dict_block = TextBlock("dict", [("label", "spam"), ("msg", "hello"), ("lang", "en"), ("skip", "x")])
    group_block = GroupBlock(
        "packet",
        [
            TextBlock("", [("label", "spam")]),
            TextBlock("", [("score", "0.9")]),
            TextBlock("", [("keep", "yes")]),
        ],
    )
    step = StepView(label="0:Example", operator_config={}, blocks=[dict_block, group_block])
    error_step = StepView(label="1:Error", operator_config={}, blocks=[], error=True)

    assert dict_block.summary() == "dict  label spam  |  msg hello  |  lang en"
    assert group_block.summary() == "packet  label spam  |  score 0.9  |  +1 more"
    assert step.summary() == "dict  label spam  |  msg hello  |  lang en  |  packet  label spam  |  score 0.9  |  +1 more"
    assert error_step.summary() == "[ERROR]"


def test_html_renderer_renders_top_level_leaf_text_block_as_single_inline_row():
    html = HtmlRenderer().render(
        [
            StepView(
                label="0:Example",
                operator_config={},
                blocks=[TextBlock("str", [("", "<generator object ...>")])],
            )
        ],
        orientation="vertical",
    )

    assert html.count('class="insp-inline-grid"') == 1
    assert "str" in html
    assert "&lt;generator object ...&gt;" in html


def test_pipeline_inspector_register_value_formatter_overrides_value_rendering():
    class _Packet:
        def __init__(self, label: str) -> None:
            self.label = label

    inspector = PipelineInspector().register_value_formatter(
        value_type=_Packet,
        formatter=lambda value: [TextBlock("packet", [("label", value.label)])],
    )

    blocks = inspector._value_to_blocks(_Packet("spam"))

    assert blocks == [TextBlock("packet", [("label", "spam")])]


def test_formatter_registry_uses_parent_value_formatter_when_local_registry_has_none():
    class _Packet:
        def __init__(self, label: str) -> None:
            self.label = label

    parent = FormatterRegistry()
    parent.register_value_formatter(
        _Packet,
        lambda value: [TextBlock("packet", [("label", value.label)])],
    )
    child = FormatterRegistry(parent=parent)

    formatter = child.find_value_formatter(_Packet)

    assert formatter is not None
    assert formatter(_Packet("spam")) == [TextBlock("packet", [("label", "spam")])]


def test_formatter_registry_prefers_exact_parent_value_formatter_before_local_subclass_formatter():
    class _PacketBase:
        def __init__(self, label: str) -> None:
            self.label = label

    class _Packet(_PacketBase):
        pass

    parent = FormatterRegistry()
    parent.register_value_formatter(
        _Packet,
        lambda value: [TextBlock("packet", [("label", value.label)])],
    )
    child = FormatterRegistry(parent=parent)
    child.register_value_formatter(
        _PacketBase,
        lambda value: [TextBlock("base", [("label", value.label)])],
    )

    formatter = child.find_value_formatter(_Packet)

    assert formatter is not None
    assert formatter(_Packet("spam")) == [TextBlock("packet", [("label", "spam")])]


def test_formatter_registry_prefers_more_specific_parent_value_formatter_before_local_object_fallback():
    class _PacketBase:
        def __init__(self, label: str) -> None:
            self.label = label

    class _Packet(_PacketBase):
        pass

    parent = FormatterRegistry()
    parent.register_value_formatter(
        _PacketBase,
        lambda value: [TextBlock("base", [("label", value.label)])],
    )
    child = FormatterRegistry(parent=parent)
    child.register_value_formatter(
        object,
        lambda value: [TextBlock("fallback", [("label", type(value).__name__)])],
    )

    formatter = child.find_value_formatter(_Packet)

    assert formatter is not None
    assert formatter(_Packet("spam")) == [TextBlock("base", [("label", "spam")])]


def test_pipeline_inspector_register_step_formatter_overrides_step_rendering():
    class _PacketOp:
        pass

    inspector = PipelineInspector().register_step_formatter(
        _PacketOp,
        lambda span, _last_image: (
            StepView(
                label=span.label,
                operator_config={"source": "custom"},
                blocks=[TextBlock("packet", [("label", str(span.output_value))])],
            ),
            None,
        ),
    )
    result = InspectionResult(
        [
            StepSpan(
                label="0:PacketOp",
                start_time=0.0,
                duration_s=0.01,
                output_value="spam",
                operator_type=_PacketOp,
            )
        ]
    )

    views = inspector.build_views(result)

    assert len(views) == 1
    assert views[0].operator_config == {"source": "custom"}
    assert views[0].blocks == [TextBlock("packet", [("label", "spam")])]


def test_formatter_registry_prefers_exact_parent_step_formatter_before_local_subclass_formatter():
    class _PacketOpBase:
        pass

    class _PacketOp(_PacketOpBase):
        pass

    parent = FormatterRegistry()
    parent.register_step_formatter(
        _PacketOp,
        lambda span, last_image: (
            StepView(
                label=span.label,
                operator_config={"source": "exact"},
                blocks=[TextBlock("packet", [("label", str(span.output_value))])],
            ),
            last_image,
        ),
    )
    child = FormatterRegistry(parent=parent)
    child.register_step_formatter(
        _PacketOpBase,
        lambda span, last_image: (
            StepView(
                label=span.label,
                operator_config={"source": "base"},
                blocks=[TextBlock("packet", [("label", str(span.output_value))])],
            ),
            last_image,
        ),
    )

    formatter = child.find_step_formatter(_PacketOp)

    assert formatter is not None
    view, last_image = formatter(
        StepSpan(
            label="0:PacketOp",
            start_time=0.0,
            duration_s=0.01,
            output_value="spam",
            operator_type=_PacketOp,
        ),
        None,
    )
    assert view.operator_config == {"source": "exact"}
    assert view.blocks == [TextBlock("packet", [("label", "spam")])]
    assert last_image is None


def test_formatter_registry_prefers_more_specific_parent_step_formatter_before_local_object_fallback():
    class _PacketOpBase:
        pass

    class _PacketOp(_PacketOpBase):
        pass

    parent = FormatterRegistry()
    parent.register_step_formatter(
        _PacketOpBase,
        lambda span, last_image: (
            StepView(
                label=span.label,
                operator_config={"source": "base"},
                blocks=[TextBlock("packet", [("label", str(span.output_value))])],
            ),
            last_image,
        ),
    )
    child = FormatterRegistry(parent=parent)
    child.register_step_formatter(
        object,
        lambda span, last_image: (
            StepView(
                label=span.label,
                operator_config={"source": "fallback"},
                blocks=[TextBlock("packet", [("label", str(span.output_value))])],
            ),
            last_image,
        ),
    )

    formatter = child.find_step_formatter(_PacketOp)

    assert formatter is not None
    view, last_image = formatter(
        StepSpan(
            label="0:PacketOp",
            start_time=0.0,
            duration_s=0.01,
            output_value="spam",
            operator_type=_PacketOp,
        ),
        None,
    )
    assert view.operator_config == {"source": "base"}
    assert view.blocks == [TextBlock("packet", [("label", "spam")])]
    assert last_image is None
