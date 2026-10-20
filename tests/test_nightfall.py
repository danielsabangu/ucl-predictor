"""Pure presentation contracts for the Nightfall design stage."""
from __future__ import annotations

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
    source = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")
    assert 'key="nf_home_print"' in source
    assert 'key="nf_home_live"' in source
    assert 'st.session_state["selected_fixture_id"] = match_identifier(first_upcoming)' in source
    assert 'st.session_state["v2_screen"] = "live"' in source
    assert 'home_can_print = FORECASTS_AVAILABLE and not predictions.empty' in source


def test_thermal_paper_is_not_restyled():
    assert ".receipt{" not in NIGHTFALL_CSS
    assert ".print-paper{" not in NIGHTFALL_CSS
    assert ".receipt{" not in NIGHTFALL_LIGHT_CSS
