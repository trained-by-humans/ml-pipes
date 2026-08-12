from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import pytest

from ml_pipes.inspection import (
    GroupBlock,
    HtmlRenderer,
    InspectionResult,
    PipelineInspector,
    StepView,
    TextBlock,
)
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


def test_pipeline_inspector_to_html_defaults_to_horizontal_orientation():
    html = PipelineInspector().to_html(_inspection_result())

    assert '<div class="insp-container insp-container--horizontal">' in html


def test_pipeline_inspector_to_html_supports_vertical_orientation():
    html = PipelineInspector().to_html(_inspection_result(), orientation="vertical")

    assert '<div class="insp-container insp-container--vertical">' in html


def test_pipeline_inspector_show_uses_browser_flow_outside_jupyter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[InspectionResult, str]] = []

    def fake_show_in_browser(self: PipelineInspector, result: InspectionResult, orientation: str = "horizontal") -> None:
        calls.append((result, orientation))

    monkeypatch.setattr("ml_pipes.inspection.inspector._IN_JUPYTER", False)
    monkeypatch.setattr(PipelineInspector, "show_in_browser", fake_show_in_browser)

    result = _inspection_result()
    PipelineInspector().show(result, cols=3, orientation="vertical")

    assert calls == [(result, "vertical")]


def test_pipeline_inspector_show_in_browser_announces_saved_report_and_browser_open(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        "ml_pipes.inspection.inspector.tempfile.mkstemp",
        _fake_mkstemp_factory(tmp_path),
    )

    opened: dict[str, str] = {}

    def fake_open(uri: str) -> bool:
        opened["uri"] = uri
        return True

    monkeypatch.setattr("ml_pipes.inspection.inspector.webbrowser.open", fake_open)

    PipelineInspector().show_in_browser(_inspection_result())
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


def test_pipeline_inspector_show_in_browser_warns_when_browser_launch_is_not_confirmed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        "ml_pipes.inspection.inspector.tempfile.mkstemp",
        _fake_mkstemp_factory(tmp_path),
    )
    monkeypatch.setattr("ml_pipes.inspection.inspector.webbrowser.open", lambda _uri: False)

    PipelineInspector().show_in_browser(_inspection_result())
    captured = capsys.readouterr()
    reports = _saved_reports(tmp_path)

    assert len(reports) == 1
    report = reports[0]
    assert report.exists()
    assert f"Inspection report saved to: {report}" in captured.err
    assert "Opening inspection report in browser..." in captured.err
    assert "Browser launch was not confirmed. If nothing opened, use the saved report path above." in captured.err


def test_html_renderer_rejects_unknown_orientation():
    with pytest.raises(ValueError, match="Invalid HTML orientation"):
        HtmlRenderer(orientation="diagonal")


def test_html_renderer_vertical_layout_widens_cards_for_tabular_output():
    html = HtmlRenderer(orientation="vertical").render(
        [StepView(label="0:Example", operator_config={}, blocks=[TextBlock("dict", [("a", "b")])])]
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


def test_html_renderer_renders_group_block_boundaries():
    html = HtmlRenderer(orientation="vertical").render(
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
        ]
    )

    assert 'class="insp-group"' in html
    assert "run: _Run" in html
    assert "stats: _Stats" in html
    assert "kept" in html


def test_html_renderer_coalesces_inline_group_rows_into_one_table():
    html = HtmlRenderer(orientation="vertical").render(
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
        ]
    )

    assert html.count('class="insp-inline-grid"') == 2
    assert "items" in html
    assert "kept" in html
    assert "matched_filter" in html


def test_pipeline_inspector_summarizes_list_of_mappings_without_generic_ellipsis():
    blocks = PipelineInspector()._value_to_blocks([{"label": "spam"}, {"label": "ham"}])

    assert blocks == [TextBlock("list  ×2", [("[0]", "dict  label spam"), ("[1]", "dict  label ham")])]


def test_html_renderer_renders_top_level_leaf_text_block_as_single_inline_row():
    html = HtmlRenderer(orientation="vertical").render(
        [
            StepView(
                label="0:Example",
                operator_config={},
                blocks=[TextBlock("str", [("", "<generator object ...>")])],
            )
        ]
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
