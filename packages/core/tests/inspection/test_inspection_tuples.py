from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from ml_pipes.inspection import GroupBlock, HtmlRenderer, ImageBlock, InspectionResult, PipelineInspector, TextBlock
from ml_pipes.tracing import StepSpan


@dataclass
class ZoneVisitMetrics:
    zone_id: int


def _views(value):
    return PipelineInspector().build_views(InspectionResult([
        StepSpan("example", 0.0, 0.01, output_value=value),
    ]))


def test_step_unwraps_only_the_outer_tuple_bundle():
    views = _views(({"detections": 1}, (ZoneVisitMetrics(0), ZoneVisitMetrics(1))))

    assert views[0].blocks == [
        GroupBlock("dict", [TextBlock("", [("detections", "1")])]),
        GroupBlock("tuple  ×2", [
            GroupBlock("[0]", [GroupBlock("ZoneVisitMetrics", [TextBlock("", [("zone_id", "0")])])]),
            GroupBlock("[1]", [GroupBlock("ZoneVisitMetrics", [TextBlock("", [("zone_id", "1")])])]),
        ]),
    ]


@pytest.mark.parametrize("orientation", ["horizontal", "vertical"])
def test_tuple_entries_support_depth_and_incremental_expansion(orientation):
    views = _views(({"detections": 1}, tuple(ZoneVisitMetrics(i) for i in range(8))))

    html = HtmlRenderer(initial_expanded_depth=1).render(views, orientation=orientation)

    assert 'tuple  ×8</summary>' in html
    assert '<details class="insp-group insp-list-item"><summary>[0] ZoneVisitMetrics</summary>' in html
    assert 'data-remaining="2"' in html
    assert '>+2 more</button>' in html
    assert '<template class="insp-list-batch" data-count="2">' in html
    assert '[7] ZoneVisitMetrics' in html
    assert views[0].blocks[1].children[-1].title == '[7]'


def test_step_bundle_preserves_individual_outputs_and_image_carry():
    image = np.zeros((4, 6, 3), dtype=np.uint8)
    result = InspectionResult([
        StepSpan("mixed", 0.0, 0.01, output_value=(image, {"detections": 1}, (ZoneVisitMetrics(0),))),
        StepSpan("next", 0.01, 0.01, output_value="done"),
    ])

    views = PipelineInspector().build_views(result)

    blocks = views[0].blocks
    assert len(blocks) == 4
    assert isinstance(blocks[0], ImageBlock)
    assert isinstance(blocks[1], TextBlock)
    assert blocks[2] == GroupBlock("dict", [TextBlock("", [("detections", "1")])])
    assert blocks[3].title == "tuple  ×1"
    assert blocks[3].children[0].title == "[0]"
    assert isinstance(views[1].blocks[0], ImageBlock)
    assert views[1].blocks[0].dim
    assert np.array_equal(views[1].blocks[0].array, image)


def test_nested_tuple_of_images_keeps_separate_indexed_previews():
    views = _views({"images": (np.zeros((4, 6, 3), dtype=np.uint8), np.ones((4, 6, 3), dtype=np.uint8))})

    html = HtmlRenderer().render(views)

    assert 'tuple  ×2' in html
    assert html.count('<img ') == 2
    assert '<summary>[0]</summary>' in html
    assert '<summary>[1]</summary>' in html


def test_tuple_field_uses_sequence_limits_instead_of_field_limits():
    views = _views({"metrics": tuple(ZoneVisitMetrics(i) for i in range(5))})

    html = HtmlRenderer(max_items_per_collection=2, max_fields_per_item=1).render(views)

    assert 'metrics: tuple  ×5</summary>' in html
    assert 'data-remaining="3"' in html
    assert '<template class="insp-list-batch" data-count="2">' in html
    assert '<template class="insp-list-batch" data-count="1">' in html
    assert '[4] ZoneVisitMetrics' in html


@pytest.mark.parametrize("value, expected", [
    ((), []),
    ((480, 640), [TextBlock("tuple", [("", "(480, 640)")])]),
])
def test_empty_and_primitive_step_tuples_keep_existing_representation(value, expected):
    assert _views(value)[0].blocks == expected


def test_empty_and_recursive_nested_tuples_keep_their_boundaries():
    assert _views({"empty": ()})[0].blocks == [GroupBlock("dict", [GroupBlock("empty: tuple  ×0", [])])]
    child = []
    value = (child,)
    child.append(value)

    views = _views(value)
    html = HtmlRenderer().render(views)

    assert views[0].blocks[0].title == "list[tuple]  ×1"
    assert '&lt;recursive tuple&gt;' in html
