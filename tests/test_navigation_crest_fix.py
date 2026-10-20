"""Navigation and future-crests tests for the Nightfall feature branch.

Runs without Streamlit because it tests isolated callback logic and source
structure. The full app and browser behaviour still require local testing.
"""
from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

from src.ui.broadcast_styles import BROADCAST_CSS

ROOT = Path(__file__).resolve().parents[1]
APP = (ROOT / "app.py").read_text(encoding="utf-8")


def _get_callbacks():
    tree = ast.parse(APP)
    names = {"_go_to_page", "_go_to_match", "_go_to_printer", "_set_feature_tab"}
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {func.name for func in functions} == names
    state = {"vault_record_id": "old", "vault_show_receipt": True, "ui_theme": "dark"}
    ns = {"st": SimpleNamespace(session_state=state), "fixture_ids": ["match-1"], "next_theme": "light"}
    module = ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[]))
    exec(compile(module, "<navigation callbacks>", "exec"), ns)
    return ns, state


def test_nav_callbacks_change_state_before_rerun():
    ns, state = _get_callbacks()
    ns["_go_to_match"]("match-1")
    assert state["v2_screen"] == "live"
    assert state["matchday_selected_id"] == "match-1"
    assert state["bp_featured_id"] == "match-1"
    ns["_go_to_page"]("vault")
    assert state["v2_screen"] == "vault"
    assert state["vault_record_id"] is None
    assert state["vault_show_receipt"] is False
    assert "toggle_ui_theme" not in APP
    assert "_toggle_theme" not in APP
    assert 'CURRENT_THEME = "dark"' in APP


def test_printer_callback_is_gated_and_resets_sharing():
    ns, state = _get_callbacks()
    ns["_go_to_printer"]("not-a-fixture")
    assert "selected_fixture_id" not in state
    ns["_go_to_printer"]("match-1")
    assert state["selected_fixture_id"] == "match-1"
    assert state["v2_screen"] == "printer"
    assert state["show_share_forecast"] is False
    ns["_set_feature_tab"]("Evidence")
    assert state["bp_feature_tab"] == "Evidence"


def test_home_and_nav_use_single_rerun_callbacks():
    home = APP.split('if current_screen == "home":', 1)[1].split('elif current_screen == "live":', 1)[0]
    nav = APP.split('current_screen = st.session_state["v2_screen"]', 1)[1].split('if current_screen == "home":', 1)[0]
    assert 'on_click=_go_to_match, args=(choice_id,)' in home
    assert 'on_click=_go_to_match, args=(selected_feature_id,)' in home
    assert 'on_click=_go_to_printer, args=(print_target,)' in home
    assert 'on_click=_go_to_page' in nav
    assert 'st.rerun()' not in home
    assert 'st.rerun()' not in "\n".join(line for line in nav.splitlines() if not line.strip().startswith("#"))


def test_external_crest_filter_is_scoped_to_future_panel():
    selector = '.bp-future-badge img.bp-club-crest{'
    assert selector in BROADCAST_CSS
    rule = BROADCAST_CSS.split(selector, 1)[1].split('}', 1)[0]
    assert 'grayscale(1)' in rule
    assert '!important' in rule
    assert '.bp-future-table{table-layout:fixed;' in BROADCAST_CSS
    assert 'bp-hero-feature img.bp-club-crest' not in BROADCAST_CSS


def test_full_app_parses_and_does_not_modify_score_generation():
    ast.parse(APP)
    assert 'build_forecast_presentation(' in APP
    assert 'build_receipt_pdf' in APP and 'build_receipt_png' in APP
    assert 'forecast_vs_reality_html(' in APP
