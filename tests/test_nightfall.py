"""Pure presentation contracts for the Nightfall design stage."""
from __future__ import annotations

import ast
import importlib
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

from src.ui.nightfall_styles import NIGHTFALL_CSS, NIGHTFALL_LIGHT_CSS, apply_nightfall_styles


def _startup(monkeypatch):
    # The test doesn't launch a Streamlit server or make network calls.
    fake_st = ModuleType("streamlit")
    fake_st.session_state = {}
    fake_st.markdown = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "streamlit", fake_st)
    sys.modules.pop("src.ui.startup_loader", None)
    return importlib.import_module("src.ui.startup_loader"), fake_st


def test_ball_loading_markup_is_compact_and_accessible(monkeypatch):
    mod, _ = _startup(monkeypatch)
    html = mod.build_startup_html()
    assert "\n" not in html
    assert "nf-splash-ball" in html
    assert 'aria-label="UCL Predictor is opening the matchday"' in html
    assert "THE LIGHTS ARE COMING ON" in html
    assert "A NIGHT OF POSSIBILITIES" in html
    assert html.count("<div") == html.count("</div>")
    assert '<svg ' in html and '</svg>' in html


def test_loader_only_renders_once(monkeypatch):
    mod, fake_st = _startup(monkeypatch)
    calls = []
    fake_st.markdown = lambda content, **kwargs: calls.append((content, kwargs))
    mod.show_startup_loading_screen()
    mod.show_startup_loading_screen()
    assert len(calls) == 1
    assert calls[0][1].get("unsafe_allow_html") is True
    assert fake_st.session_state[mod.STARTUP_SESSION_KEY] is True


def test_nightfall_theme_switching():
    renders = []
    apply_nightfall_styles(renders.append, theme="dark")
    assert len(renders) == 1
    assert "#071329" in renders[0]
    assert "NIGHTFALL_LIGHT_CSS" not in renders[0]
    renders.clear()
    apply_nightfall_styles(renders.append, theme="light")
    assert "#f3f8ff" in renders[0]
    assert "nightfall_home_actions" in renders[0]
    assert "@media(prefers-reduced-motion:reduce)" in renders[0]


def test_visual_code_does_not_request_live_data():
    root = Path(__file__).resolve().parents[1]
    for name in ("src/ui/nightfall_styles.py", "src/ui/startup_loader.py"):
        text = (root / name).read_text(encoding="utf-8")
        assert "api-football.com/" not in text
        assert "realtimesportsapi.com" not in text


def test_home_has_two_working_primary_actions():
    """Check the real button callbacks, not a removed inline click handler.

    Navigation now happens in on_click callbacks before Streamlit rerenders.
    Requiring an old inline assignment would reject the correct implementation.
    """
    source = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")
    tree = ast.parse(source)

    button_callbacks = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not (isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "st"
                and node.func.attr == "button"):
            continue
        options = {kw.arg: kw.value for kw in node.keywords if kw.arg is not None}
        key_node = options.get("key")
        callback_node = options.get("on_click")
        if (isinstance(key_node, ast.Constant)
                and isinstance(key_node.value, str)
                and isinstance(callback_node, ast.Name)):
            button_callbacks[key_node.value] = callback_node.id

    assert button_callbacks.get("nf_home_print") == "_go_to_printer"
    assert button_callbacks.get("nf_home_live") == "_go_to_match"
    assert 'home_can_print = FORECASTS_AVAILABLE and not predictions.empty' in source

    callbacks = {
        node.name: node for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }
    assert "_go_to_printer" in callbacks
    assert "_go_to_match" in callbacks
    printer_body = ast.unparse(callbacks["_go_to_printer"])
    live_body = ast.unparse(callbacks["_go_to_match"])
    assert "st.session_state['selected_fixture_id'] = fixture_id" in printer_body
    assert "st.session_state['v2_screen'] = 'printer'" in printer_body
    assert "st.session_state['v2_screen'] = 'live'" in live_body
    assert "st.session_state['matchday_selected_id'] = choice_id" in live_body


def test_thermal_paper_is_not_restyled():
    assert ".receipt{" not in NIGHTFALL_CSS
    assert ".print-paper{" not in NIGHTFALL_CSS
    assert ".receipt{" not in NIGHTFALL_LIGHT_CSS
