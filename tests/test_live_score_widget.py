"""Regression tests for the optional official Champions League scoreboard embed.

These tests exercise only the function extracted from app.py; unlike importing a
Streamlit application, they do not run the prediction pipeline or read credentials.
"""
from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

APP = Path(__file__).resolve().parents[1] / "app.py"


def _subject():
    tree = ast.parse(APP.read_text(encoding="utf-8"))
    node = next(
        item for item in tree.body
        if isinstance(item, ast.FunctionDef) and item.name == "render_ucl_live_scoreboard"
    )
    html_calls = []
    headers = []
    captions = []
    environment = {
        "render_html": headers.append,
        "components": SimpleNamespace(html=lambda markup, **kw: html_calls.append((markup, kw))),
        "st": SimpleNamespace(caption=captions.append),
    }
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(APP), "exec"), environment)
    environment["render_ucl_live_scoreboard"]()
    return tree, headers, html_calls, captions


def test_provider_widget_is_official_champions_league_embed():
    _, headers, html_calls, captions = _subject()
    assert len(headers) == len(html_calls) == len(captions) == 1
    markup, config = html_calls[0]
    assert 'class="rsa-scoreboard"' in markup
    assert 'data-league="champions-league"' in markup
    assert 'src="https://www.realtimesportsapi.com/widget.js"' in markup
    assert 'https://www.realtimesportsapi.com/embed/scoreboard/champions-league' in markup
    assert config["height"] >= 600
    assert config["scrolling"] is True


def test_widget_requires_no_api_credentials_or_server_requests():
    _, _, html_calls, _ = _subject()
    markup, _ = html_calls[0]
    assert 'api/v1' not in markup
    assert 'Authorization' not in markup
    assert 'Bearer ' not in markup
    assert 'st.secrets' not in markup
    assert 'fetch(' not in markup


def test_native_score_is_optional_and_original_forecasts_are_unchanged():
    source = APP.read_text(encoding="utf-8")
    assert 'render_selected_match_score(' in source
    assert 'match_fixture_score(centre_row, events)' in source
    assert 'score=match_score' in source
    assert 'build_forecast_presentation(' in source
    assert 'predictions = filter_upcoming_predictions(predictions)' in source


def test_widget_only_renders_on_live_page():
    tree = ast.parse(APP.read_text(encoding="utf-8"))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == 'render_ucl_live_scoreboard']
    assert len(calls) == 1
    live_branch = [node for node in tree.body
                   if isinstance(node, ast.If)
                   and ast.unparse(node.test) == "current_screen == 'home'"]
    # The screens form an if/elif chain. The live screen is its first elif.
    assert len(live_branch) == 1
    assert live_branch[0].orelse and isinstance(live_branch[0].orelse[0], ast.If)
    assert ast.unparse(live_branch[0].orelse[0].test) == "current_screen == 'live'"
    assert any(node in ast.walk(live_branch[0].orelse[0]) for node in calls)


def test_visible_attribution_and_non_live_disclaimer():
    _, headers, _, captions = _subject()
    assert 'SCOREBOARD' in headers[0]
    assert 'Third-party scoreboard' in captions[0]
    assert 'pre-match forecasts never update' in captions[0]
