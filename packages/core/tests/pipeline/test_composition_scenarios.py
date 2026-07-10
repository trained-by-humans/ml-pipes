"""
Real-world composition scenarios testing isolation and context sharing
across embed() and inline() in realistic multi-stage pipelines.
"""
import pytest

from ml_pipes.core import (
    Pipeline,
    embed,
    inline,
)
from ml_pipes.standard import (
    Store,
    Recall,
    Pick,
)
from ml_pipes.validation import PipelineValidationError
from ml_pipes.context import ContextOp
from typing import Any


# ---------------------------------------------------------------------------
# Shared operator stubs
# ---------------------------------------------------------------------------

class Double:
    def __call__(self, value: int) -> int:
        return value * 2


class AddOne:
    def __call__(self, value: int) -> int:
        return value + 1


class IntToString:
    def __call__(self, value: int) -> str:
        return str(value)


class StringToInt:
    def __call__(self, value: str) -> int:
        return int(value)


class Negate:
    def __call__(self, value: int) -> int:
        return -value


class ContextSnapshot(ContextOp):
    """Records the outer context keys at the point it runs."""
    def __init__(self, target: dict):
        self.target = target

    def apply(self, current, context):
        self.target.update(context.values)
        return current, context

    def resolve_contract(self, current_output, stored_annotations, expand, err):
        return (Any,), Any if current_output is None else current_output


# ---------------------------------------------------------------------------
# embed: two independent embeds must not share context
# ---------------------------------------------------------------------------

def test_two_embeds_with_internal_stores_do_not_bleed_into_each_other():
    # Each embed stores under the same key — the second should not see the first's value.
    sub_a = Pipeline([Store("result"), Double()])
    sub_b = Pipeline([Store("result"), AddOne()])

    snapshot = {}
    outer = Pipeline([
        embed(sub_a),
        embed(sub_b),
        ContextSnapshot(snapshot),
    ])

    outer(10)

    # Neither inner Store should have leaked into the outer context.
    assert "result" not in snapshot


def test_outer_store_before_embed_is_available_after_embed():
    # The outer pipeline stores a value, runs an embed, then recalls it —
    # the embed must not wipe or see the outer stored value.
    sub = Pipeline([Double()])
    snapshot_before = {}
    snapshot_after = {}

    outer = Pipeline([
        Store("original"),
        ContextSnapshot(snapshot_before),
        embed(sub),
        ContextSnapshot(snapshot_after),
    ])

    outer(5)

    assert snapshot_before == {"original": 5}
    assert snapshot_after == {"original": 5}  # unchanged by embed


def test_outer_recall_after_embed_retrieves_correct_value():
    sub = Pipeline([Double()])

    outer = Pipeline([
        Store("pre"),
        embed(sub),
        Recall("pre"),
        Pick(1),  # get the recalled value
    ])

    result = outer(4)
    assert result == 4  # original value, not the doubled one


def test_nested_embed_isolates_at_both_levels():
    # inner_inner stores a key; inner embeds inner_inner; outer embeds inner.
    # Nothing should leak to the outer context.
    inner_inner = Pipeline([Store("deep")])
    inner = Pipeline([embed(inner_inner)])
    snapshot = {}

    outer = Pipeline([
        embed(inner),
        ContextSnapshot(snapshot),
    ])

    outer(1)
    assert "deep" not in snapshot


def test_same_pipeline_embedded_twice_has_no_shared_state():
    # Embedding the same pipeline object twice — each run must be independent.
    call_count = {"n": 0}

    class CountedDouble:
        def __call__(self, value: int) -> int:
            call_count["n"] += 1
            return value * 2

    sub = Pipeline([Store("val"), CountedDouble()])
    snapshot = {}

    outer = Pipeline([
        embed(sub),
        embed(sub),
        ContextSnapshot(snapshot),
    ])

    result = outer(3)

    # Both embeds ran (value doubled twice = 12)
    assert result == 12
    assert call_count["n"] == 2
    # Neither Store leaked out
    assert "val" not in snapshot


# ---------------------------------------------------------------------------
# inline: context is shared — Store/Recall work across the boundary
# ---------------------------------------------------------------------------

def test_inline_store_before_boundary_recalled_inside_inlined_pipeline():
    sub = Pipeline([Recall("seed")])

    outer = Pipeline([
        Store("seed"),
        inline(sub),
    ])

    result = outer(7)
    assert result == (7, 7)


def test_inline_store_inside_inlined_pipeline_recalled_after_boundary():
    sub = Pipeline([Store("mid")])

    outer = Pipeline([
        inline(sub),
        Recall("mid"),
        Pick(1),
    ])

    result = outer(9)
    assert result == 9


def test_two_inlines_in_sequence_share_the_same_context():
    sub_a = Pipeline([Store("from_a")])
    sub_b = Pipeline([Recall("from_a"), Pick(1)])

    outer = Pipeline([
        inline(sub_a),
        inline(sub_b),
    ])

    result = outer(3)
    assert result == 3


def test_inline_internal_keys_coexist_with_outer_keys():
    # Outer stores "outer_key"; inlined pipeline stores "inner_key".
    # Both should be visible afterwards.
    sub = Pipeline([Store("inner_key")])
    snapshot = {}

    outer = Pipeline([
        Store("outer_key"),
        inline(sub),
        ContextSnapshot(snapshot),
    ])

    outer(42)

    assert snapshot["outer_key"] == 42
    assert snapshot["inner_key"] == 42


def test_outer_store_survives_across_multiple_inlines():
    sub_a = Pipeline([Double()])
    sub_b = Pipeline([AddOne()])
    snapshot = {}

    outer = Pipeline([
        Store("original"),
        inline(sub_a),
        inline(sub_b),
        ContextSnapshot(snapshot),
    ])

    outer(5)

    # original stored before any inline — must still be there
    assert snapshot["original"] == 5


# ---------------------------------------------------------------------------
# embed vs inline: direct contrast in the same pipeline
# ---------------------------------------------------------------------------

def test_embed_and_inline_side_by_side_correct_isolation():
    # Store a value, run an embed (should not see it), run an inline (should see it).
    sub = Pipeline([Recall("key")])
    snapshot_embed = {}
    snapshot_inline = {}

    outer = Pipeline([
        Store("key"),
        embed(Pipeline([ContextSnapshot(snapshot_embed)])),
        inline(Pipeline([ContextSnapshot(snapshot_inline)])),
    ])

    outer(1)

    assert "key" not in snapshot_embed   # embed: isolated
    assert snapshot_inline["key"] == 1   # inline: shared


def test_embed_after_inline_does_not_see_inline_stored_values():
    # Inline stores a key; subsequent embed must not see it.
    sub_inline = Pipeline([Store("shared")])
    snapshot = {}
    sub_embed = Pipeline([ContextSnapshot(snapshot)])

    outer = Pipeline([
        inline(sub_inline),
        embed(sub_embed),
    ])

    outer(0)

    assert "shared" not in snapshot
