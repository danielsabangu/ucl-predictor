"""Regression tests for the dark-only and phone-sized Matchday Board UI."""
from pathlib import Path
import ast

ROOT = Path(__file__).resolve().parents[1]
APP = (ROOT / "app.py").read_text(encoding="utf-8")
CSS = (ROOT / "src/ui/broadcast_styles.py").read_text(encoding="utf-8")


def test_no_light_mode_control_or_callback():
    ast.parse(APP)
    assert 'CURRENT_THEME = "dark"' in APP
    assert 'st.session_state["ui_theme"] = "dark"' in APP
    assert 'st.columns(5, gap="small")' in APP
    assert 'toggle_ui_theme' not in APP
    assert '_toggle_theme' not in APP
    assert 'theme_button_label' not in APP


def test_responsive_board_uses_full_width_fixture_and_action():
    assert 'max-width: 820px' in CSS
    assert '[class*="st-key-bp_fixture_item_"]' in CSS
    assert 'flex-direction: column !important' in CSS
    assert 'overflow-wrap: normal !important' in CSS
    assert 'height: 34px !important' in CSS
    assert 'width: 100% !important' in CSS
    assert 'grid-template-columns: 45px minmax(0,1fr) 58px !important' in CSS


def test_nav_on_phone_has_four_tabs():
    assert ':nth-child(4){display:none!important}' in CSS
    assert 'width:25%!important' in CSS


def test_broadcast_light_theme_is_inaccessible():
    from src.ui.broadcast_styles import apply_broadcast_styles
    captured = []
    apply_broadcast_styles(captured.append, theme="light")
    assert len(captured) == 1
    assert "#eef5fd" not in captured[0]
    assert "#061324" in captured[0]


def test_printer_and_provenance_kept():
    assert '"PRINT FORECAST"' in APP
    assert '"OPEN MATCH CENTRE"' in APP
    assert '"VIEW MATCH"' in APP
    assert 'on_click=_go_to_printer' in APP
    assert 'build_receipt_pdf' in APP and 'build_receipt_png' in APP
    assert 'forecast_vs_reality_html' in APP
