"""Regression tests for the permission-gated Match Centre event timeline.

No network requests, provider credentials or Streamlit server are required.
"""
from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

from src.ui.live_event_view import event_timeline_html, parse_match_moments


APP = Path(__file__).resolve().parents[1] / "app.py"


def _event_gate(environ: dict[str, str]) -> bool:
    tree = ast.parse(APP.read_text(encoding="utf-8"))
    node = next(
        item for item in tree.body
        if isinstance(item, ast.FunctionDef)
        and item.name == "_live_event_display_allowed"
    )
    namespace = {
        "os": SimpleNamespace(environ=environ),
        "st": SimpleNamespace(secrets={}),
    }
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(APP), "exec"), namespace)
    return namespace["_live_event_display_allowed"]()


def test_timeline_off_by_default():
    assert _event_gate({}) is False


def test_timeline_requires_both_explicit_flags():
    assert _event_gate({"LIVE_EVENT_TIMELINE_ENABLED": "1"}) is False
    assert _event_gate({"LIVE_EVENT_DISPLAY_AUTHORIZED": "1"}) is False
    assert _event_gate({
        "LIVE_EVENT_TIMELINE_ENABLED": "1",
        "LIVE_EVENT_DISPLAY_AUTHORIZED": "true",
    }) is True


def test_unknown_event_never_becomes_a_goal_from_scoring_flag():
    assert parse_match_moments([
        {"id": "p1", "type": "unverified", "scoringPlay": True, "text": "Goal?"}
    ]) == []


def test_known_provider_labels_are_preserved_without_prediction_inference():
    moments = parse_match_moments([
        {"id": "g1", "type": "goal", "clock": "57:22",
         "text": "A verified goal", "team": {"name": "Home"}},
        {"id": "c1", "type": "yellow card", "minute": 60,
         "text": "A verified yellow card", "team": {"name": "Away"}},
        {"id": "s1", "type": "substitution", "clock": "72:01",
         "text": "A verified substitution"},
    ])
    assert [m.kind for m in moments] == ["GOAL", "YELLOW CARD", "SUBSTITUTION"]
    assert [m.minute for m in moments] == ["57'", "60'", "72'"]
    assert moments[0].team == "Home"


def test_duplicate_plays_not_repeated():
    plays = [{"id": "same", "type": "goal", "clock": "14:00", "text": "Goal"}] * 2
    assert len(parse_match_moments(plays)) == 1


def test_timeline_escapes_provider_content():
    moment = parse_match_moments([{
        "id": "g2", "type": "goal",
        "text": '<script>alert("x")</script>',
        "team": {"name": "<img src=x onerror=alert(1)>"},
    }])[0]
    markup = event_timeline_html([moment], state="no_data")
    assert "<script>" not in markup
    assert "<img src=x" not in markup
    assert "&lt;script&gt;" in markup
    assert "&lt;img" in markup


def test_empty_and_disabled_states_are_honest():
    markup = event_timeline_html([], state="disabled")
    assert "not enabled for public display" in markup
    assert "0 recorded moments" not in markup


def test_selected_fixture_uses_tested_adapter_and_cache_only_when_authorized():
    app = APP.read_text(encoding="utf-8")
    assert "if _live_event_display_allowed():" in app
    assert "match_fixture_event_id(centre_row.to_dict(), provider_events)" in app
    assert "_cached_live_plays(key, event_id)" in app
    assert "fetch_match_plays(api_key, event_id, limit=50)" in app
    assert "render_html(event_timeline_html(moments, state=state))" in app
    assert "@st.cache_data(ttl=600, show_spinner=False)" in app
