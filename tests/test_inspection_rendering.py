from __future__ import annotations

from dataclasses import dataclass

import pytest

from ml_pipes import (
    GroupBlock,
    HtmlRenderer,
    InspectionResult,
    PipelineInspector,
    StepSpan,
    StepView,
    TextBlock,
)


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


def test_pipeline_inspector_to_html_defaults_to_horizontal_orientation():
    html = PipelineInspector().to_html(_inspection_result())

    assert '<div class="insp-container insp-container--horizontal">' in html


def test_pipeline_inspector_to_html_supports_vertical_orientation():
    html = PipelineInspector().to_html(_inspection_result(), orientation="vertical")

    assert '<div class="insp-container insp-container--vertical">' in html


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
    blocks = PipelineInspector()._output_to_blocks(_Run(stats=_Stats(scanned=0, kept=1), started_at=2.5))

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
    blocks = PipelineInspector()._output_to_blocks({"stats": {"kept": 1}, "started_at": 2.5})

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
    blocks = PipelineInspector()._output_to_blocks([{"label": "spam"}, {"label": "ham"}])

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
